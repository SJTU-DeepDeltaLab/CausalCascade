import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


IMAGE_NAME = "first_frame.png"
PROMPT_NAME = "prompt.txt"


@dataclass(frozen=True)
class Case:
    case_id: str
    series: str
    scene: str
    scene_dir: Path
    image_path: Path
    prompt_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def counterfactual_seed_case_id(case_id: str) -> str:
    """Map an edited case to its factual scene for paired seeding."""
    parts = case_id.split(".")
    if (
        len(parts) >= 4
        and parts[0] in {"move_type", "scale_type", "delete_type", "mass_type"}
        and parts[-1].startswith("edit_")
    ):
        return ".".join(parts[1:-1])
    return case_id


def stable_case_seed(base_seed: int, case_id: str, modulo: int = 1_000_000_000) -> int:
    """Return the existing deterministic seed shared by factual and edited cases."""
    digest = sha256_text(f"{base_seed}:{counterfactual_seed_case_id(case_id)}")
    return int(digest[:8], 16) % modulo


def choose_seed(args: Any, case: Case, idx: int, rng: random.Random) -> int:
    """Select a seed for runners using the shared benchmark seed policy."""
    if args.random_seed:
        return rng.randint(0, 1_000_000_000)
    if args.seed_strategy == "case_hash":
        return stable_case_seed(args.seed, case.case_id)
    if args.seed_strategy == "sequential":
        return args.seed + idx - 1
    if args.seed_strategy == "same":
        return args.seed
    raise ValueError(f"Unknown seed strategy: {args.seed_strategy}")


def safe_name(value: str) -> str:
    bad = '<>:"/\\|?*'
    return "".join("_" if c in bad else c for c in value)


def parse_gpu_list(value: Optional[str]) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def collect_cases(dataset_dir: Path, only_series: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Case]:
    dataset_dir = dataset_dir.resolve()
    selected = set(only_series or [])
    cases: List[Case] = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    edit_types = {"move_type", "scale_type", "delete_type", "mass_type"}
    candidates = []
    for top in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if top.name in edit_types:
            for series_dir in sorted(p for p in top.iterdir() if p.is_dir()):
                for scene_dir in sorted(p for p in series_dir.iterdir() if p.is_dir()):
                    for edit_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("edit_")):
                        candidates.append((series_dir.name, scene_dir.name, edit_dir,
                                           f"{top.name}.{series_dir.name}.{scene_dir.name}.{edit_dir.name}"))
        else:
            for scene_dir in sorted(p for p in top.iterdir() if p.is_dir()):
                candidates.append((top.name, scene_dir.name, scene_dir,
                                   f"{top.name}.{scene_dir.name}"))

    for series, scene, scene_dir, case_id in candidates:
        if selected and series not in selected:
            continue
        image_path = scene_dir / IMAGE_NAME
        prompt_path = scene_dir / PROMPT_NAME
        missing = []
        if not image_path.exists():
            missing.append(IMAGE_NAME)
        if not prompt_path.exists():
            missing.append(PROMPT_NAME)
        if missing:
            logging.warning("Skipping %s: missing %s", scene_dir, ", ".join(missing))
            continue
        cases.append(
            Case(
                case_id=case_id,
                series=series,
                scene=scene,
                scene_dir=scene_dir,
                image_path=image_path,
                prompt_path=prompt_path,
            )
        )
        if limit is not None and len(cases) >= limit:
            return cases
    return cases


def case_video_path(output_dir: Path, case: Case) -> Path:
    return output_dir / f"{safe_name(case.case_id)}.mp4"


def case_frames_dir(output_dir: Path, case: Case) -> Path:
    return output_dir / "frames" / safe_name(case.case_id)


def read_prompt(case: Case) -> str:
    return case.prompt_path.read_text(encoding="utf-8").strip()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "created_at_utc": utc_now(), "runs": []}


def base_metadata(
    model_name: str,
    model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    args: Any,
    backend: str,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "model_dir": str(model_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "backend": backend,
        "command": sys.argv,
        "args": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
    }


def case_metadata(case: Case, prompt: str, video_path: Path, frames_dir: Optional[Path]) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "series": case.series,
        "scene": case.scene,
        "scene_dir": str(case.scene_dir.resolve()),
        "first_frame_path": str(case.image_path.resolve()),
        "prompt_path": str(case.prompt_path.resolve()),
        "first_frame_sha256": sha256_file(case.image_path),
        "prompt_sha256": sha256_text(prompt),
        "output_video": str(video_path.resolve()),
        "output_frames_dir": str(frames_dir.resolve()) if frames_dir else None,
    }


def append_manifest(manifest_path: Path, run_record: Dict[str, Any]) -> None:
    manifest = load_manifest(manifest_path)
    manifest["updated_at_utc"] = utc_now()
    manifest.setdefault("runs", []).append(run_record)
    write_json(manifest_path, manifest)


def export_frames_from_video(video_path: Path, frames_dir: Path, fps: Optional[int] = None) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
    except Exception:
        logging.warning("imageio is unavailable; skipping frame export for %s", video_path)
        return 0

    count = 0
    for count, frame in enumerate(iio.imiter(video_path), start=1):
        out = frames_dir / f"frame_{count:04d}.png"
        iio.imwrite(out, frame)
    return count


def run_subprocess(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    logging.info("Running command: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def record_failure(exc: BaseException) -> Dict[str, str]:
    return {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
