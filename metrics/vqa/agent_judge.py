#!/usr/bin/env python3
"""GT-visible, per-question VQA agent runner.

The preferred input is a JSONL manifest produced by
``prepare_agentic_model_level_manifest.py``.
Every row carries the exact generated-video and ground-truth-video paths, which
prevents the cross-scene GT capture bug in the original reproduction script.
The reference answer is used only for scoring and is never sent to the judge.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

R = os.path.dirname(os.path.abspath(__file__))

_BALANCED_PROMPT_PATH = os.path.join(R, "balanced_prompt_v2.txt")
with open(_BALANCED_PROMPT_PATH, encoding="utf-8") as _fh:
    SYSTEM_PROMPT = _fh.read()

def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120]


def _load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def load_manifest(path):
    required = {
        "unit_id", "scene_id", "prompt_level", "model_name", "gen_video", "gt_video",
        "question_id", "question", "answer_options", "ground_truth_answer",
        "diagnostic_dimension", "diagnostic_subdimension",
    }
    tasks = _load_jsonl(path)
    for i, task in enumerate(tasks, 1):
        missing = sorted(required - set(task))
        if missing:
            raise ValueError(f"manifest row {i} missing fields: {missing}")
    return tasks


def load_legacy_questions(vqa_dir, video_dir, gt_dir):
    """Load the old per-scene 697-question layout for audit reproduction."""
    tasks = []
    for fn in sorted(os.listdir(vqa_dir)):
        if not fn.endswith(".json") or "summary" in fn or "answers" in fn:
            continue
        payload = json.load(open(os.path.join(vqa_dir, fn)))
        video_file = payload.get("video_file") or fn.replace(".json", ".mp4")
        for q in payload.get("answers", []):
            scene = payload["scene"]
            tasks.append({
                "unit_id": f"legacy::{scene}::{q['question_id']}",
                "scene_id": scene,
                "prompt_level": "legacy",
                "model_name": os.path.basename(os.path.abspath(video_dir)),
                "gen_video": os.path.join(video_dir, video_file),
                "gt_video": os.path.join(gt_dir, video_file),
                "question_id": q["question_id"],
                "question": q["question"],
                "answer_options": q.get("answer_options", []),
                "ground_truth_answer": q.get("ground_truth_answer"),
                "diagnostic_dimension": q.get("diagnostic_dimension"),
                "diagnostic_subdimension": q.get("diagnostic_subdimension"),
                "source_event_ids": q.get("source_event_ids", []),
                "source_edge_ids": q.get("source_edge_ids", []),
            })
    return tasks


def _normalize_answer(answer, options):
    if answer is None:
        return None
    # Providers occasionally omit only the final sentence punctuation from an
    # otherwise verbatim option.  Treat that typographic difference as equal,
    # while keeping the semantic option match exact.
    value = str(answer).strip().rstrip(" .;:").lower()
    for option in options:
        if value == str(option).strip().rstrip(" .;:").lower():
            return option
    return None


def _input_signature(task):
    fields = [
        task.get("unit_id"), os.path.realpath(task.get("gen_video", "")),
        os.path.realpath(task.get("gt_video", "")), task.get("question"),
        task.get("answer_options"), task.get("ground_truth_answer"),
    ]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _consistency_label(votes, expected_votes):
    """Classify answer stability across independent judge calls.

    For the planned three-vote protocol, 3/3 agreement is high consistency,
    a 2/3 majority is medium consistency, and any unresolved disagreement or
    invalid answer is low consistency.  The conservative invalid-answer rule
    prevents a missing/ill-formed vote from being hidden by a majority.
    """
    if expected_votes != 3:
        return "not_computed"
    if len(votes) != 3 or any(v.get("predicted") is None for v in votes):
        return "low"
    counts = Counter(v["predicted"] for v in votes)
    top = counts.most_common()
    if top[0][1] == 3:
        return "high"
    if top[0][1] == 2:
        return "medium"
    return "low"


def _review_priority(votes, consistency):
    """Turn consistency and self-reported confidence into a review queue.

    This is a triage signal, not a correctness estimate: a unanimous but
    confidently wrong answer can still pass through the low-priority queue.
    """
    confidences = [v.get("confidence") for v in votes]
    if consistency == "low" or "low" in confidences:
        return "urgent"
    if consistency == "medium" or "medium" in confidences:
        return "recommended"
    return "audit_only"


def _uncertainty_level(consistency):
    """Expose the user-facing uncertainty interpretation of consistency."""
    return {
        "high": "low",
        "medium": "medium",
        "low": "high",
        "not_computed": "not_available",
    }.get(consistency, "not_available")


def _consistency_summary(records):
    labels = Counter(r.get("consistency_confidence", "missing") for r in records)
    priorities = Counter(r.get("review_priority", "missing") for r in records)
    uncertainty = Counter(r.get("uncertainty_level", "missing") for r in records)
    return {
        "vote_count": Counter(r.get("vote_count", 0) for r in records),
        "consistency_confidence": dict(sorted(labels.items())),
        "uncertainty_level": dict(sorted(uncertainty.items())),
        "review_priority": dict(sorted(priorities.items())),
        "manual_review_recommended": sum(
            bool(r.get("manual_review_recommended")) for r in records),
    }


def _aggregate_votes(task, votes, expected_votes):
    """Create one final record from a complete set of vote-level records."""
    usable = [v["predicted"] for v in votes if v.get("predicted") is not None]
    counts = Counter(usable)
    ranked = counts.most_common()
    has_strict_majority = bool(ranked) and (
        len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    if has_strict_majority:
        majority, majority_n = ranked[0]
        final = dict(next(v for v in votes if v.get("predicted") == majority))
    else:
        majority, majority_n = None, 0
        final = dict(votes[0])
    consistency = _consistency_label(votes, expected_votes)
    priority = _review_priority(votes, consistency)
    final["predicted"] = majority
    final["is_correct"] = (
        majority == task["ground_truth_answer"] if majority is not None else None)
    final["votes"] = [v.get("predicted") for v in votes]
    final["vote_confidences"] = [v.get("confidence") for v in votes]
    final["majority_n"] = majority_n
    final["vote_count"] = len(votes)
    final["vote_answer_counts"] = dict(counts)
    final["consistency_confidence"] = consistency
    final["uncertainty_level"] = _uncertainty_level(consistency)
    final["review_priority"] = priority
    final["manual_review_recommended"] = priority != "audit_only"
    final.pop("vote_index", None)
    return final


def _distribution(tasks):
    def counts(field):
        return dict(sorted(Counter(t.get(field) for t in tasks).items(), key=lambda x: str(x[0])))
    return {
        "units": len(tasks),
        "scenes": len({t["scene_id"] for t in tasks}),
        "prompt_levels": counts("prompt_level"),
        "models": counts("model_name"),
        "dimensions": counts("diagnostic_dimension"),
        "subdimensions": counts("diagnostic_subdimension"),
    }


def _score_summary(records):
    valid = [r for r in records if r.get("is_correct") is not None]

    def grouped(field):
        groups = defaultdict(list)
        for rec in valid:
            groups[rec.get(field)].append(rec)
        return {
            str(k): {
                "n": len(v),
                "correct": sum(bool(x["is_correct"]) for x in v),
                "accuracy": sum(bool(x["is_correct"]) for x in v) / len(v),
            }
            for k, v in sorted(groups.items(), key=lambda x: str(x[0]))
        }

    return {
        "completed": len(records),
        "scored": len(valid),
        "correct": sum(bool(r["is_correct"]) for r in valid),
        "accuracy": (sum(bool(r["is_correct"]) for r in valid) / len(valid)) if valid else None,
        "by_dimension": grouped("diagnostic_dimension"),
        "by_model": grouped("model_name"),
        "by_prompt_level": grouped("prompt_level"),
    }


def _usage_summary(records):
    """Aggregate provider usage and cache telemetry saved by agent_client."""
    totals = Counter()
    rounds = 0
    tool_calls = 0
    shared_context_records = 0
    for record in records:
        usage = record.get("usage_totals") or {}
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
        rounds += int(record.get("n_rounds") or 0)
        tool_calls += len(record.get("tools_used") or [])
        shared_context_records += int(bool(record.get("shared_context_used")))
    # The Anthropic-compatible endpoint currently returns
    # ``cache_read_input_tokens``/``cache_creation_input_tokens`` and
    # ``input_tokens``; the OpenAI-compatible endpoint uses the
    # ``prompt_cache_*`` names.  Normalize both forms for the report.
    hit = totals.get("prompt_cache_hit_tokens", 0) + totals.get("cache_read_input_tokens", 0)
    if "prompt_cache_miss_tokens" in totals:
        miss = totals.get("prompt_cache_miss_tokens", 0)
    else:
        miss = totals.get("input_tokens", 0) + totals.get("cache_creation_input_tokens", 0)
    return {
        "records_with_usage": sum(bool(r.get("usage_totals")) for r in records),
        "api_sessions": len(records),
        "api_rounds": rounds,
        "tool_calls": tool_calls,
        "shared_context_records": shared_context_records,
        "usage_totals": dict(totals),
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "prompt_cache_hit_rate": (hit / (hit + miss)) if (hit + miss) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="JSONL rollout-question manifest")
    source.add_argument("--vqa_dir", help="legacy per-scene vqa_results directory")
    parser.add_argument("--video_dir", help="required with --vqa_dir")
    parser.add_argument("--gt_dir", help="required with --vqa_dir")
    parser.add_argument("--mode", default="B", choices=["A", "B"],
                        help="B exposes GT and alignment tools; A is generated-video only")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scenes", default="")
    parser.add_argument("--qids", default="")
    parser.add_argument("--unit_ids", default="",
                        help="comma-separated manifest unit_ids (useful for targeted prompt audits)")
    parser.add_argument("--unit_ids_file", type=Path, default=None,
                        help="newline-delimited manifest unit_ids for targeted reruns")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--shared-context", action="store_true",
                        help="prepend the cached GEN/GT contact-sheet context to each question")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.vqa_dir and (not args.video_dir or not args.gt_dir):
        parser.error("--video_dir and --gt_dir are required with --vqa_dir")

    sys.path.insert(0, R)
    from agent_client import run_agent
    from tools_api import Ctx

    tasks = (load_manifest(args.manifest) if args.manifest else
             load_legacy_questions(args.vqa_dir, args.video_dir, args.gt_dir))
    scene_filter = {x for x in args.scenes.split(",") if x}
    qid_filter = {x for x in args.qids.split(",") if x}
    unit_filter = {x for x in args.unit_ids.split(",") if x}
    if args.unit_ids_file:
        unit_filter.update(
            x.strip() for x in args.unit_ids_file.read_text().splitlines() if x.strip()
        )
    if scene_filter:
        tasks = [t for t in tasks if t["scene_id"] in scene_filter]
    if qid_filter:
        tasks = [t for t in tasks if t["question_id"] in qid_filter]
    if unit_filter:
        tasks = [t for t in tasks if t["unit_id"] in unit_filter]
    if args.limit:
        tasks = tasks[:args.limit]

    missing = []
    for task in tasks:
        for field in ("gen_video", "gt_video"):
            if not os.path.isfile(task[field]):
                missing.append({"unit_id": task["unit_id"], "field": field, "path": task[field]})
    if missing:
        print(json.dumps({"missing_files": missing[:20], "missing_count": len(missing)}, indent=2))
        raise SystemExit(2)

    print(json.dumps(_distribution(tasks), indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    outdir = os.path.join(R, "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "agent_answers.jsonl")
    votes_path = os.path.join(outdir, "agent_votes.jsonl")
    if args.redo:
        suffix = time.strftime("%Y%m%d-%H%M%S")
        for path in (out_path, votes_path):
            if os.path.exists(path):
                os.replace(path, path + f".bak-{suffix}")

    cache = {}
    task_by_id = {task["unit_id"]: task for task in tasks}
    current_signatures = {task["unit_id"]: _input_signature(task) for task in tasks}
    existing_records = []
    if os.path.exists(out_path):
        existing_records = _load_jsonl(out_path)
        for rec in existing_records:
            unit_id = rec.get("unit_id")
            task = task_by_id.get(unit_id)
            valid_answer = (_normalize_answer(rec.get("predicted"), task["answer_options"])
                            if task else None)
            vote_fields_ready = (
                args.votes == 1 or
                (len(rec.get("votes", [])) == args.votes and
                 "consistency_confidence" in rec and
                 "review_priority" in rec)
            )
            if (unit_id in current_signatures
                    and rec.get("input_sha256") == current_signatures[unit_id]
                    and valid_answer is not None and vote_fields_ready):
                rec["predicted"] = valid_answer
                cache[unit_id] = rec

    # If a process was interrupted after writing vote-level records but before
    # appending the aggregate answer, recover that case without making another
    # set of API calls.  Only a complete vote set for the current task is used.
    if args.votes > 1 and os.path.exists(votes_path):
        vote_groups = defaultdict(list)
        for vote in _load_jsonl(votes_path):
            if vote.get("unit_id") in task_by_id:
                vote_groups[vote["unit_id"]].append(vote)
        for unit_id, grouped_votes in vote_groups.items():
            if unit_id in cache or len(grouped_votes) < args.votes:
                continue
            task = task_by_id[unit_id]
            latest_votes = grouped_votes[-args.votes:]
            if all(v.get("input_sha256") == current_signatures[unit_id]
                   for v in latest_votes):
                cache[unit_id] = _aggregate_votes(task, latest_votes, args.votes)
    if existing_records and len(existing_records) != len(cache):
        suffix = time.strftime("%Y%m%d-%H%M%S")
        os.replace(out_path, out_path + f".bak-invalid-or-stale-{suffix}")
        with open(out_path, "w") as fh:
            for task in tasks:
                if task["unit_id"] in cache:
                    fh.write(json.dumps(cache[task["unit_id"]], ensure_ascii=False) + "\n")
        print(f"retained {len(cache)} valid cached answers; "
              f"will rerun {len(tasks) - len(cache)} invalid/stale units", flush=True)

    pending = [t for t in tasks if t["unit_id"] not in cache]
    print(f"pending: {len(pending)} / {len(tasks)}", flush=True)
    write_lock = threading.Lock()

    def run_once(task, vote_index):
        identity = hashlib.sha256(
            json.dumps([task["unit_id"], task["gen_video"], task["gt_video"]]).encode()
        ).hexdigest()[:16]
        ctx = Ctx(f"{_safe_name(task['scene_id'])}__{identity}", task["gen_video"],
                  task["gt_video"], mode=args.mode)
        shared_context = None
        if args.shared_context:
            from tools_api import build_shared_context
            shared_context = build_shared_context(ctx)
        transcript = []
        verdict = run_agent(task["question"], task["answer_options"], ctx, SYSTEM_PROMPT,
                            max_rounds=args.max_rounds, transcript=transcript,
                            dimension=task.get("diagnostic_dimension"),
                            subdimension=task.get("diagnostic_subdimension"),
                            prior_answer=task.get("prior_answer"),
                            shared_context=shared_context)
        predicted_raw = verdict.get("answer")
        predicted = _normalize_answer(predicted_raw, task["answer_options"])
        transcript_dir = os.path.join(outdir, "transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        transcript_name = (
            f"{_safe_name(task['unit_id'])[:80]}__{identity}__vote{vote_index + 1:02d}.json")
        transcript_path = os.path.join(transcript_dir, transcript_name)
        with write_lock:
            with open(transcript_path, "w") as fh:
                json.dump(transcript, fh, indent=2, ensure_ascii=False)
        return {
            **task,
            "input_sha256": _input_signature(task),
            "mode": args.mode,
            "predicted_raw": predicted_raw,
            "predicted": predicted,
            "is_correct": (predicted == task["ground_truth_answer"]) if predicted else None,
            "confidence": verdict.get("confidence"),
            "reasoning": (verdict.get("reasoning") or "")[:1200],
            "tools_used": verdict.get("tools_used", []),
            "trail": verdict.get("trail", []),
            "n_rounds": verdict.get("n_rounds"),
            "requested_model": verdict.get("requested_model"),
            "evaluator_model_label": verdict.get("evaluator_model_label"),
            "served_models": verdict.get("served_models", []),
            "usage_rounds": verdict.get("usage_rounds", []),
            "usage_totals": verdict.get("usage_totals", {}),
            "shared_context_used": verdict.get("shared_context_used", False),
            "transcript_path": transcript_path,
            "error": predicted is None,
            "vote_index": vote_index,
        }

    def work(task):
        try:
            votes = [run_once(task, i) for i in range(args.votes)]
        except Exception as exc:
            # A transient provider/tool failure must not abort the entire
            # ThreadPool.  Persist an invalid vote so that a resumed run can
            # discard it and retry only this unit.
            votes = [{
                **task,
                "input_sha256": _input_signature(task),
                "mode": args.mode,
                "predicted_raw": None,
                "predicted": None,
                "is_correct": None,
                "confidence": "low",
                "reasoning": "worker exception: " + str(exc)[:500],
                "tools_used": [], "trail": [], "n_rounds": 0,
                "requested_model": None, "evaluator_model_label": None,
                "served_models": [], "usage_rounds": [], "usage_totals": {},
                "shared_context_used": bool(args.shared_context),
                "transcript_path": None, "error": True,
                "error_message": str(exc)[:1000], "vote_index": 0,
            }]
        final = _aggregate_votes(task, votes, args.votes)
        with write_lock:
            with open(votes_path, "a") as fh:
                for vote in votes:
                    fh.write(json.dumps(vote, ensure_ascii=False) + "\n")
            with open(out_path, "a") as fh:
                fh.write(json.dumps(final, ensure_ascii=False) + "\n")
        return final

    completed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(work, task) for task in pending]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 5 == 0 or completed == len(pending):
                print(f"[{completed}/{len(pending)}] {time.time() - started:.0f}s", flush=True)

    records_by_id = dict(cache)
    if os.path.exists(out_path):
        for rec in _load_jsonl(out_path):
            records_by_id[rec["unit_id"]] = rec
    ordered = [records_by_id[t["unit_id"]] for t in tasks if t["unit_id"] in records_by_id]
    summary = {
        "tag": args.tag,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metric_definition": ("accuracy is the generated rollout's answer match to the "
                              "ground-truth rollout; it is not evaluator-vs-human agreement"),
        "input_distribution": _distribution(tasks),
        "uncertainty_protocol": {
            "independent_votes_per_case": args.votes,
            "consistency_definition": {
                "high": "3/3 normalized answers agree",
                "medium": "2/3 normalized answers agree",
                "low": "no strict majority or at least one invalid answer",
            },
            "uncertainty_level": {
                "low": "3/3 agreement",
                "medium": "2/3 agreement",
                "high": "no strict majority or at least one invalid answer",
            },
            "review_priority_definition": {
                "urgent": "low consistency or any vote reports low confidence",
                "recommended": "medium consistency or any vote reports medium confidence",
                "audit_only": "high consistency and all votes report high confidence",
            },
        },
        "uncertainty_summary": _consistency_summary(ordered),
        "usage_summary": _usage_summary(ordered),
        "shared_context_enabled": bool(args.shared_context),
        **_score_summary(ordered),
    }
    with open(os.path.join(outdir, "agent_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
