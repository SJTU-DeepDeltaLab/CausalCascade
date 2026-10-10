#!/usr/bin/env python3
"""Materialize released HF inputs for inference and evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath


def relative_id(value: str, parts: int | None = None) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe repository path: {value}")
    if parts is not None and len(path.parts) != parts:
        raise ValueError(f"Expected {parts} path components: {value}")
    return Path(*path.parts)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def materialize(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination) or digest(source) == digest(destination):
            return
        if not overwrite:
            raise FileExistsError(f"Different file already exists: {destination}")
    temporary = destination.with_name(destination.name + ".prepare.tmp")
    if temporary.exists():
        raise FileExistsError(f"Remove stale preparation file before retrying: {temporary}")
    try:
        if mode == "hardlink":
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
        else:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text(path: Path, text: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        if not overwrite:
            raise FileExistsError(f"Different text already exists: {path}")
    path.write_text(text, encoding="utf-8")


def prepare(base: list[dict], pairs: list[dict], fetch, output: Path,
            mode: str = "copy", overwrite: bool = False) -> dict:
    by_scene = {row["scene_id"]: row for row in base}
    if len(by_scene) != len(base):
        raise ValueError("Duplicate base scene IDs")
    variants = set()
    for row in pairs:
        variant = relative_id(row["variant_id"], 4)
        if row["scene_id"] not in by_scene or row["variant_id"] in variants:
            raise ValueError(f"Unknown scene or duplicate variant: {row['variant_id']}")
        variants.add(row["variant_id"])
        if variant.parts[0] not in {"delete_type", "mass_type", "move_type", "scale_type"}:
            raise ValueError(f"Unknown edit type: {variant.parts[0]}")
        if "/".join(variant.parts[1:3]) != row["scene_id"]:
            raise ValueError(f"Variant does not match its scene: {row['variant_id']}")
        factual = by_scene[row["scene_id"]]
        for kind in ("initial_frame", "reference_video"):
            if row[f"factual_{kind}_path"] != factual[f"{kind}_path"]:
                raise ValueError(f"Mismatched factual {kind}: {row['variant_id']}")

    def copy(relative: str, target: Path) -> None:
        relative_id(relative)
        materialize(fetch(relative), output / target, mode, overwrite)

    for row in base:
        scene_id = relative_id(row["scene_id"], 2)
        reference = Path("reference") / scene_id
        copy(row["reference_video_path"], reference / f"{scene_id.name}.mp4")
        copy(row["reference_video_path"], Path("reference_flat") / (".".join(scene_id.parts) + ".mp4"))
        for field, name in (("event_graph_path", "event_graph.json"), ("qa_path", "diagnostic_vqa.json"),
                            ("scene_metadata_path", f"{scene_id.name}.json")):
            copy(row[field], reference / name)
        copy(row["initial_frame_path"], reference / "first_frame.png")
        for level in range(1, 4):
            target = Path("inputs") / f"L{level}" / scene_id
            copy(row["initial_frame_path"], target / "first_frame.png")
            write_text(output / target / "prompt.txt", row[f"prompt_level{level}"], overwrite)

    for row in pairs:
        target = Path("counterfactual") / relative_id(row["variant_id"], 4)
        for field, name in (("counterfactual_initial_frame_path", "first_frame.png"),
                            ("counterfactual_reference_video_path", "video.mp4"),
                            ("delta_json_path", "diff.json"), ("counterfactual_prompt_path", "prompt.txt")):
            copy(row[field], target / name)
        if (output / target / "prompt.txt").read_text(encoding="utf-8").strip() != row["counterfactual_prompt_level1"].strip():
            raise ValueError(f"Prompt table/file mismatch: {row['variant_id']}")
    return {"base_scenes": len(base), "rq2_pairs": len(pairs),
            "edit_counts": dict(sorted(Counter(row["edit_type"] for row in pairs).items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="deepdeltalab/CausalCascade")
    parser.add_argument("--revision", default="main", help="Resolved once to an HF commit")
    parser.add_argument("--source-root", type=Path, help="Use a local HF-format dataset instead of downloading")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--link-mode", choices=("copy", "hardlink"), default="copy",
                        help="Treat hardlinked data as read-only")
    parser.add_argument("--overwrite", action="store_true", help="Replace conflicting prepared files")
    args = parser.parse_args()
    import pyarrow.parquet as pq

    if args.source_root:
        source = args.source_root.resolve()
        if args.output_dir.resolve().is_relative_to(source):
            raise ValueError("Output must not be inside the source dataset")
        revision = None

        def fetch(relative: str) -> Path:
            path = (source / relative_id(relative)).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise FileNotFoundError(f"Missing or unsafe source file: {relative}")
            return path
    else:
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().dataset_info(args.repo_id, revision=args.revision).sha

        def fetch(relative: str) -> Path:
            relative_id(relative)
            return Path(hf_hub_download(repo_id=args.repo_id, repo_type="dataset",
                                       revision=revision, filename=relative))

    def table(name: str) -> list[dict]:
        data = pq.read_table(fetch(f"tables/{name}.parquet"))
        fields = [name for name in data.column_names if name.endswith("_path") or name.endswith("_id")
                  or name.startswith("prompt_level") or name in {"edit_type", "counterfactual_prompt_level1"}]
        return data.select(fields).to_pylist()

    base, pairs = table("base_scenes"), table("rq2_pairs")
    counts = Counter(row["edit_type"] for row in pairs)
    if len(base) != 50 or counts != {"delete": 50, "mass": 75, "move": 50, "scale": 50}:
        raise ValueError(f"Unexpected coverage: {len(base)} scenes, {dict(counts)} edits")
    report = prepare(base, pairs, fetch, args.output_dir.resolve(), args.link_mode, args.overwrite)
    report.update({"repo_id": args.repo_id if not args.source_root else None,
                   "revision": revision, "link_mode": args.link_mode})
    write_text(args.output_dir / "preparation.json", json.dumps(report, indent=2) + "\n", args.overwrite)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
