#!/usr/bin/env python3
"""Score final agentic and deterministic answers using the paper's macro rule."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dataset_questions import load_questions


DIMENSIONS = (
    "event_occurrence", "causal_pathway", "temporal_ordering",
    "outcome_termination", "scene_consistency",
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def indexed(rows: list[dict], description: str) -> dict[str, dict]:
    output = {row["unit_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Duplicate unit_id in {description}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentic-manifest", required=True, type=Path)
    parser.add_argument("--agentic-answers", required=True, type=Path)
    parser.add_argument("--scene-manifest", required=True, type=Path)
    parser.add_argument("--scene-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="Dataset root containing <series>/<scene>/diagnostic_vqa.json")
    parser.add_argument("--questions", type=Path, default=None,
                        help="Optional legacy flattened question JSONL")
    args = parser.parse_args()

    agent_manifest = indexed(load_jsonl(args.agentic_manifest), "agentic manifest")
    scene_manifest = indexed(load_jsonl(args.scene_manifest), "scene manifest")
    agent = indexed(load_jsonl(args.agentic_answers), "agentic answers")
    payload = json.loads(args.scene_predictions.read_text(encoding="utf-8"))
    scene = indexed(payload["predictions"], "scene predictions")
    if set(agent) != set(agent_manifest) or set(scene) != set(scene_manifest):
        raise ValueError("Answer IDs must exactly match their manifests")
    if set(agent) & set(scene):
        raise ValueError("Agentic and deterministic unit IDs overlap")
    if any(task["diagnostic_dimension"] == "scene_consistency" for task in agent_manifest.values()):
        raise ValueError("Scene Consistency questions must use the deterministic branch")
    if any(task["diagnostic_dimension"] != "scene_consistency" for task in scene_manifest.values()):
        raise ValueError("Agentic questions must use the agentic branch")

    bank_rows, _, _ = load_questions(args.questions, args.dataset_root)
    bank = {(q["scene_id"], q["question_id"]): q for q in bank_rows}
    if len(bank) != len(bank_rows):
        raise ValueError("Duplicate scene/question ID in question bank")
    manifest_rows = list(agent_manifest.values()) + list(scene_manifest.values())
    manifest_keys = {(q["scene_id"], q["question_id"]) for q in manifest_rows}
    if len(manifest_keys) != len(manifest_rows) or manifest_keys != set(bank):
        raise ValueError("Manifests must cover the canonical question bank exactly once")
    question_fields = (
        "question", "answer_type", "answer_options", "ground_truth_answer",
        "ground_truth_answer_code", "diagnostic_dimension", "diagnostic_subdimension",
        "source_event_ids", "source_edge_ids",
    )
    for task in manifest_rows:
        reference = bank[task["scene_id"], task["question_id"]]
        if any(task.get(field) != reference.get(field) for field in question_fields):
            raise ValueError(f"Manifest question differs from canonical bank: {task['unit_id']}")

    grouped = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: {"correct": 0, "questions": 0})
    for source, predictions, answer_field in (
        (agent_manifest, agent, "agentic"),
        (scene_manifest, scene, "deterministic"),
    ):
        for uid, task in source.items():
            result = predictions[uid]
            if answer_field == "agentic":
                if result.get("adaptive_status") == "no_majority_manual_review":
                    raise ValueError(f"Unresolved agentic vote: {uid}")
                answer = result.get("adaptive_final", result.get("predicted"))
            else:
                answer = result.get("predicted_answer")
            if answer not in task["answer_options"]:
                raise ValueError(f"Invalid {answer_field} answer: {uid}")
            correct = int(answer == task["ground_truth_answer"])
            dim = task["diagnostic_dimension"]
            grouped[dim][task["scene_id"]].append(correct)
            counts[dim]["correct"] += correct
            counts[dim]["questions"] += 1

    if set(grouped) != set(DIMENSIONS):
        raise ValueError("Expected all five diagnostic dimensions")
    expected = {dim: sum(q["diagnostic_dimension"] == dim for q in bank_rows) for dim in DIMENSIONS}
    if ({dim: counts[dim]["questions"] for dim in DIMENSIONS} != expected
            or len({scene_id for scenes in grouped.values() for scene_id in scenes}) != 50):
        raise ValueError("Evaluation does not cover the complete canonical question bank")
    dimension_scores = {
        dim: sum(sum(correct) / len(correct) for correct in grouped[dim].values())
        / len(grouped[dim])
        for dim in DIMENSIONS
    }
    total = sum(item["questions"] for item in counts.values())
    report = {
        "protocol": "exact match; mean within scene/dimension, then scene macro, then five-dimension macro",
        "question_count": total,
        "scene_count": len({scene_id for scenes in grouped.values() for scene_id in scenes}),
        "dimension_macro_scores": dimension_scores,
        "overall_dimension_macro": sum(dimension_scores.values()) / len(DIMENSIONS),
        "exact_question_accuracy": sum(item["correct"] for item in counts.values()) / total,
        "counts": {dim: counts[dim] for dim in DIMENSIONS},
        "source_manifests": {"agentic": str(args.agentic_manifest),
                             "scene_consistency": str(args.scene_manifest)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "overall_dimension_macro": report["overall_dimension_macro"],
                      "question_count": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
