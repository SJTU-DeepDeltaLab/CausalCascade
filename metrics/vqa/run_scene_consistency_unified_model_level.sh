#!/usr/bin/env bash
set -euo pipefail

# Efficient full-model deterministic run. SAM2 is executed once per unique
# scene (the four Scene Consistency questions for a scene share the same
# video), then the evidence is keyed to each of the four question units. The
# rule projection remains unchanged.

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd)}"
NAME="$1"
MANIFEST="$2"
OUT="$3"
MODEL_ID="${4:-facebook/sam2.1-hiera-tiny}"
# Optional explicit CPU affinity. With no fifth argument, the host/cgroup
# scheduler chooses available cores; previous cluster runs passed 0-127.
CORES="${5:-}"
PIN=()
if [[ -n "$CORES" ]]; then
  command -v taskset >/dev/null || { echo "taskset is required for CPU affinity" >&2; exit 1; }
  PIN=(taskset -c "$CORES")
fi
REPRO="$ROOT"
mkdir -p "$OUT/logs" "$OUT/work"

python3 - "$MANIFEST" "$OUT/work/unique_scene_manifest.jsonl" <<'PY'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
seen = set(); rows = []
for line in src.open():
    row = json.loads(line)
    if row.get("diagnostic_dimension") != "scene_consistency":
        continue
    scene = row["scene_id"]
    if scene in seen:
        continue
    seen.add(scene); rows.append(row)
with dst.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
assert len(rows) == 50, len(rows)
print("unique scenes", len(rows))
PY

env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 "${PIN[@]}" \
  python3 "$REPRO/scene_consistency_sam2_tracker.py" \
    --manifest "$OUT/work/unique_scene_manifest.jsonl" --max-frames 13 --max-objects 16 \
    --sparse-redetection --reference-activity --model-id "$MODEL_ID" --device cpu \
    --output "$OUT/unified_sam2_evidence.jsonl" >"$OUT/logs/unified.log" 2>&1

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${PIN[@]}" \
  python3 "$REPRO/extract_scene_consistency_evidence.py" \
    --manifest "$OUT/work/unique_scene_manifest.jsonl" --scale 0.20 \
    --output "$OUT/unified_static_evidence.jsonl" >"$OUT/logs/static.log" 2>&1

python3 - "$MANIFEST" "$OUT/unified_sam2_evidence.jsonl" "$OUT/unified_static_evidence.jsonl" "$OUT" <<'PY'
import copy, json, sys
from pathlib import Path
manifest, sam_path, static_path, out = map(Path, sys.argv[1:])
tasks = [json.loads(x) for x in manifest.open() if x.strip()]
sam_by_scene = {row["scene_id"]: row for row in
                (json.loads(x) for x in sam_path.open() if x.strip())}
static_by_scene = {row["scene_id"]: row for row in
                   (json.loads(x) for x in static_path.open() if x.strip())}
by_sub = {"entity_persistence": [], "identity_geometry_integrity": [],
          "trajectory_continuity": [], "static_scene_camera_stability": []}
for task in tasks:
    scene = task["scene_id"]; sub = task["diagnostic_subdimension"]
    row = copy.deepcopy(static_by_scene[scene] if sub == "static_scene_camera_stability"
                        else sam_by_scene[scene])
    row.update({"unit_id": task["unit_id"], "subdimension": sub,
                "model_name": task.get("model_name"),
                "prompt_level": task.get("prompt_level"), "scene_id": scene})
    by_sub[sub].append(row)
names = {"entity_persistence":"entity_evidence.jsonl",
         "identity_geometry_integrity":"identity_evidence.jsonl",
         "trajectory_continuity":"trajectory_evidence.jsonl",
         "static_scene_camera_stability":"static_evidence.jsonl"}
for sub, rows in by_sub.items():
    path = out / names[sub]
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert len(rows) == 50, (sub, len(rows))
print({sub: len(rows) for sub, rows in by_sub.items()})
PY

python3 "$REPRO/project_scene_consistency_exact.py" \
  --manifest "$MANIFEST" \
  --entity-evidence "$OUT/entity_evidence.jsonl" \
  --identity-evidence "$OUT/identity_evidence.jsonl" \
  --trajectory-evidence "$OUT/trajectory_evidence.jsonl" \
  --static-evidence "$OUT/static_evidence.jsonl" \
  --expected-n 200 --output "$OUT/scene_consistency_200_exact_predictions.json"

python3 - "$OUT" "$NAME" <<'PY'
import json, sys
from pathlib import Path
out, name = Path(sys.argv[1]), sys.argv[2]
pred = json.loads((out / "scene_consistency_200_exact_predictions.json").read_text())
counts = {}
for row in pred["predictions"]:
    sub = row["diagnostic_subdimension"]
    key = str(row["predicted_option"])
    counts.setdefault(sub, {})[key] = counts.setdefault(sub, {}).get(key, 0) + 1
summary = {"name": name, "n": pred["n"],
           "evidence_mode": "one_unified_13frame_16object_pass_per_scene",
           "option_counts_by_subdimension": counts,
           "labels_loaded": pred["labels_loaded"]}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY
