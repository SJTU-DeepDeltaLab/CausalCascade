from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_FIELDS = (
    "initial_frame_path",
    "factual_initial_frame_path",
    "counterfactual_initial_frame_path",
)
VIDEO_FIELDS = (
    "reference_video_path",
    "factual_reference_video_path",
    "counterfactual_reference_video_path",
)
ANNOTATION_FIELDS = {
    "event_graph_path": "event_graph",
    "qa_path": "diagnostic_qa",
    "scene_metadata_path": "scene_metadata",
    "delta_json_path": "delta",
}


class CausalCascadeDataset:
    """Resolve CausalCascade media and annotations lazily per sample."""

    def __init__(
        self,
        repo_id: str | None = None,
        subset: str = "base_scenes",
        split: str = "test",
        *,
        local_root: str | Path | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        decode_images: bool = True,
        decode_video: bool = False,
        video_size: tuple[int, int] | None = None,
        video_frame_stride: int = 1,
        video_max_frames: int | None = None,
        load_annotations: bool = True,
    ) -> None:
        if (repo_id is None) == (local_root is None):
            raise ValueError("Provide exactly one of repo_id or local_root")

        from datasets import load_dataset

        self.repo_id = repo_id
        self.local_root = Path(local_root).resolve() if local_root is not None else None
        self.revision = revision
        self.token = token
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.decode_images = decode_images
        self.decode_video = decode_video
        if video_frame_stride < 1:
            raise ValueError("video_frame_stride must be positive")
        if video_max_frames is not None and video_max_frames < 1:
            raise ValueError("video_max_frames must be positive")
        self.video_size = video_size
        self.video_frame_stride = video_frame_stride
        self.video_max_frames = video_max_frames
        self.load_annotations = load_annotations

        source = str(self.local_root) if self.local_root is not None else repo_id
        local_options = {}
        if self.local_root is not None:
            from datasets import Features
            import pyarrow.parquet as pq

            table_names = {
                "base_scenes": "base_scenes",
                "counterfactuals": "counterfactual_variants",
                "rq2_pairs": "rq2_pairs",
                "diagnostic_qa": "diagnostic_questions",
            }
            if subset in table_names:
                schema = pq.read_schema(
                    self.local_root / "tables" / f"{table_names[subset]}.parquet"
                )
                local_options["features"] = Features.from_arrow_schema(schema)
        self.table = load_dataset(
            source,
            subset,
            split=split,
            revision=revision,
            token=token,
            cache_dir=self.cache_dir,
            **local_options,
        )
        # The wrapper decodes from paths; native media columns serve the Hub viewer.
        preview_columns = [
            field.removesuffix("_path") for field in IMAGE_FIELDS + VIDEO_FIELDS
            if field.removesuffix("_path") in self.table.column_names
        ]
        if preview_columns:
            self.table = self.table.remove_columns(preview_columns)

    def __len__(self) -> int:
        return len(self.table)

    def _resolve(self, relative_path: str) -> Path:
        if self.local_root is not None:
            path = (self.local_root / relative_path).resolve()
            if not path.is_relative_to(self.local_root):
                raise ValueError(f"Path escapes dataset root: {relative_path}")
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=relative_path,
                revision=self.revision,
                token=self.token,
                cache_dir=self.cache_dir,
            )
        )

    @staticmethod
    def _decode_image(path: Path):
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB")).copy()
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    @staticmethod
    def _decode_video(
        path: Path,
        size: tuple[int, int] | None,
        frame_stride: int,
        max_frames: int | None,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Video decoding requires the optional dependency: pip install -e '.[video]'"
            ) from exc
        import numpy as np
        import torch

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        frames = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_stride:
                    frame_index += 1
                    continue
                frame_index += 1
                if size is not None:
                    height, width = size
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            capture.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from: {path}")
        array = np.stack(frames)
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.table[index])

        if self.decode_images:
            for field in IMAGE_FIELDS:
                value = sample.get(field)
                if value:
                    sample[field.removesuffix("_path")] = self._decode_image(self._resolve(value))

        if self.decode_video:
            for field in VIDEO_FIELDS:
                value = sample.get(field)
                if value:
                    sample[field.removesuffix("_path")] = self._decode_video(
                        self._resolve(value),
                        self.video_size,
                        self.video_frame_stride,
                        self.video_max_frames,
                    )

        if self.load_annotations:
            for field, output_field in ANNOTATION_FIELDS.items():
                value = sample.get(field)
                if value:
                    sample[output_field] = json.loads(
                        self._resolve(value).read_text(encoding="utf-8-sig")
                    )
        return sample


def collate_fn(batch: list[Any]) -> Any:
    """Stack equal-shaped tensors and preserve variable-length values as lists."""
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, dict):
        keys = set().union(*(item.keys() for item in batch))
        return {key: collate_fn([item.get(key) for item in batch]) for key in sorted(keys)}
    if all(item is None for item in batch):
        return None
    if any(item is None for item in batch):
        return batch
    try:
        import torch
        if all(isinstance(item, torch.Tensor) for item in batch):
            if all(item.shape == first.shape for item in batch):
                return torch.stack(batch)
            return batch
        if all(isinstance(item, (int, float, bool)) for item in batch):
            return torch.tensor(batch)
    except ImportError:
        pass
    return batch
