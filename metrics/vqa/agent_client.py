#!/usr/bin/env python3
"""Multi-round vision client for an Anthropic-compatible API endpoint."""
import base64, io, json, os, time, copy
import requests
from PIL import Image

from agent_runtime_utils import _load_env_file, _parse_json

_load_env_file()
BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic").rstrip("/")
MODEL = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-flash-vision-exp[1m]")
MODEL_LABEL = os.environ.get("EVALUATOR_MODEL_LABEL", "DeepSeek-V4.1-Flash")
TOKEN = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
MAX_TOKENS = 32768
# The pilot revealed that the difficult cases are usually not language-generation
# failures but missed event boundaries.  Give the judge enough internal budget to
# inspect a second focused crop and reconcile it with the literal event checklist.
THINKING = {"type": "enabled", "budget_tokens": int(
    os.environ.get("DEEPSEEK_THINKING_BUDGET", "24576"))}
# Keep a fixed temperature within a run, but allow an explicitly selected
# independent-vote protocol to use a little sampling diversity.  This is still
# one prompt and one decision rule; it is not an ensemble of prompts.
TEMPERATURE = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.2"))


def _flatten_usage(usage):
    """Flatten numeric usage fields while retaining provider-specific names."""
    flat = {}
    if not isinstance(usage, dict):
        return flat
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = flat.get(key, 0) + value
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (int, float)) and not isinstance(nested_value, bool):
                    name = f"{key}.{nested_key}"
                    flat[name] = flat.get(name, 0) + nested_value
    return flat


def _sum_usage(usages):
    total = {}
    for usage in usages:
        for key, value in _flatten_usage(usage).items():
            total[key] = total.get(key, 0) + value
    return total


def declare_tools(mode, tools_doc):
    spec = []
    for name, t in MODE_TOOLS(mode).items():
        schema = copy.deepcopy(t["schema"])
        if mode == 'A' and "video" in schema.get("properties", {}):
            schema["properties"]["video"]["enum"] = ["gen"]
            schema["properties"]["video"]["default"] = "gen"
        spec.append({"name": name, "description": t["desc"], "input_schema": schema})
    return spec


def MODE_TOOLS(mode):
    from tools_api import TOOLS
    return {n: t for n, t in TOOLS.items() if mode == 'B' or t['mode'] != 'B'}


def ask_round(msgs, tools_spec, system_prompt=None, timeout=120, retries=3):
    if not TOKEN:
        raise RuntimeError("DEEPSEEK_API_KEY (or ANTHROPIC_AUTH_TOKEN) is not configured")
    body = {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "thinking": THINKING, "messages": msgs}
    if system_prompt:
        body["system"] = [{"type": "text", "text": system_prompt}]
    if tools_spec:
        body["tools"] = tools_spec
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{BASE}/v1/messages",
                              headers={"Authorization": f"Bearer {TOKEN}",
                                       "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json=body, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code >= 400:
                error_text = r.text[:500]
                # Preserve the normal 641-style image payload by default.  A
                # compact retry is activated only for an explicit provider
                # image-format/size rejection; it is not a second prompt or
                # an ensemble decision.
                if (attempt == 0 and _is_image_request_error(error_text)):
                    # The fallback is deliberately conservative: contact sheets
                    # are normally sent at the canonical resolution, but when a
                    # provider rejects a tool image we reduce every image in
                    # the current conversation enough to stay below the
                    # provider's aggregate image-size limit.
                    _compact_image_messages(msgs, max_width=1024, quality=60)
                    body["messages"] = msgs
                    continue
                raise ValueError(f"{r.status_code}: {error_text[:300]}")
            return r.json()
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    if last is None:
        # A run of 429 responses leaves no Python exception in ``last``;
        # raising None would obscure the actual provider-side rate limit.
        last = RuntimeError("provider request retries exhausted (likely rate limit)")
    print(f"  [warn] ask_round exhausted retries: {str(last)[:90]}", flush=True)
    raise last


def _is_image_request_error(text):
    lowered = (text or "").lower()
    return (
        "unsupported image" in lowered
        or "image" in lowered and ("size" in lowered or "format" in lowered)
        or "tool_result.image" in lowered
    )


def _compact_image_messages(messages, max_width=1680, quality=75):
    """Compact image blocks in-place, used only after an image request error."""
    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        if value.get("type") == "image":
            source = value.get("source") or {}
            data = source.get("data")
            if source.get("type") == "base64" and data:
                try:
                    raw = base64.b64decode(data, validate=True)
                    with Image.open(io.BytesIO(raw)) as im:
                        if im.width > max_width:
                            im = im.resize(
                                (max_width, int(im.height * max_width / im.width)),
                                Image.BILINEAR,
                            )
                        buf = io.BytesIO()
                        im.convert("RGB").save(
                            buf, "JPEG", quality=quality, optimize=True
                        )
                    source["media_type"] = "image/jpeg"
                    source["data"] = base64.b64encode(buf.getvalue()).decode("ascii")
                except Exception:
                    # Leave an image untouched if it cannot be decoded here;
                    # the original provider error will be reported normally.
                    pass
        for child in value.values():
            visit(child)

    visit(messages)


def run_agent(question, options, ctx, system_prompt, max_rounds=6, verbose=False,
              transcript=None, dimension=None, subdimension=None, prior_answer=None,
              shared_context=None):
    """One question, one session. Returns verdict dict with tool trail."""
    mode = ctx.mode
    tools_spec = declare_tools(mode, None)
    focus = ""
    if dimension == "temporal_ordering":
        focus = ("Dimension focus: verify the visible onset and completion of BOTH named events "
                 "before choosing an order; if either event is incomplete or absent, choose the "
                 "explicit absence option. Treat every named direction, count, repetition, and "
                 "endpoint as required; do not infer an event from a later state.\n")
    elif dimension == "causal_pathway":
        focus = ("Dimension focus: establish event presence first, then look for a visible direct "
                 "contact/support/transfer. A generic related motion or partial compound event is "
                 "not enough; mere temporal precedence is insufficient for a direct edge.\n")
    elif dimension == "outcome_termination":
        focus = ("Dimension focus: search the entire generated timeline for the first moment the "
                 "stated terminal conjunction is complete, then inspect later frames for replay or regression.\n")
    elif subdimension == "trajectory_continuity":
        focus = ("Dimension focus: judge the named object's visual path and identity, compensating for "
                 "camera/view shifts; tracker fragments, flow spikes, and background corruption are "
                 "not semantic discontinuities unless a visible object-level jump or unsupported "
                 "reappearance is confirmed.\n")
    elif subdimension == "static_scene_camera_stability":
        focus = ("Dimension focus: use background/anchored landmarks to decide camera motion and inspect "
                 "anchored objects separately; foreground motion does not imply a camera shift.\n")
    elif subdimension == "entity_persistence":
        focus = ("Dimension focus: count visually identifiable instances over time; distinguish expected "
                 "leaving the frame from unexplained disappearance, and require a visible extra instance "
                 "or unsupported identity change for duplication.\n")
    prior_hint = (f"A previous draft answer (hypothesis only, not ground truth) was: {prior_answer}. "
                  "Recheck it against the generated video and overturn it when the evidence conflicts.\n"
                  if prior_answer else "")
    question_text = (
        f"QUESTION: {question}\n"
        f"ANSWER OPTIONS (reply verbatim, exactly one): {' | '.join(map(str, options))}\n"
        f"{prior_hint}"
        f"{focus}"
        f"Evidence budget: {max_rounds} tool rounds max; aim to ANSWER by round "
        f"{min(max_rounds, 9)}. First inspect both whole videos, then gather focused "
        f"visual observations at the event onset, endpoint, and (for terminal/persistence "
        f"questions) the final fifth of the generated clip. In mode B, use `paired_zoom` "
        f"for at least one difficult event boundary so GEN and GT are seen at the same times. "
        f"Use generic flow/tracks only "
        f"after visual evidence has been collected and only to resolve a remaining ambiguity.\n"
        f"Important: generic track/flow statistics are only supporting evidence. Do not "
        f"treat a fragment, spawn, or failed alignment as a semantic object failure "
        f"without a matching visual crop. For a relation, verify both endpoints before "
        f"ordering or assigning causality. Before selecting an option, silently complete "
        f"a checklist for every named event: present/absent, participant identity, initial "
        f"condition, direction, required contact or repeated/count endpoint, and completion. "
        f"For a relation, only compare order after both checklists pass.\n"
        f"Tool protocol: to gather evidence call one tool at a time. "
        f"When ready, answer ONLY with a JSON object:\n"
        f'{{"answer": "<one option>", "confidence": "high|medium|low", "reasoning": "2-3 sentences", '
        f'"evidence_tools": [{{"tool": "<name>", "claimed_fact": "..."}}]}}')
    if shared_context:
        question_text += (
            "\nA stable visual context was supplied before this question. Treat those GEN/GT "
            "contact sheets as the initial whole-video observation; do not call `observe` "
            "again unless a different sampling is needed.\n"
        )
    initial_content = list(shared_context or [])
    initial_content.append({"type": "text", "text": question_text})
    msgs = [{"role": "user", "content": initial_content}]
    trail = []
    served_models = []
    usage_rounds = []
    for r in range(max_rounds):
        final_turn = r == max_rounds - 1
        round_tools = [] if final_turn else tools_spec
        if final_turn:
            reminder = ({"type": "text", "text":
                         "The evidence-gathering budget is now exhausted. Do not call any more "
                         "tools. Return only the required final JSON object with exactly one of "
                         "the supplied answer options."})
            if msgs and msgs[-1].get("role") == "user" and isinstance(msgs[-1].get("content"), list):
                msgs[-1]["content"].append(reminder)
            else:
                msgs.append({"role": "user", "content": [reminder]})
        t0 = time.time()
        data = ask_round(msgs, round_tools, system_prompt=system_prompt)
        usage = data.get("usage") or {}
        usage_rounds.append({"round": r, "usage": usage})
        if data.get("model") and data.get("model") not in served_models:
            served_models.append(data["model"])
        blocks = data.get("content", [])
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        texts = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if verbose:
            print(f"  round{r}: blocks={[b['type'] for b in blocks]}", flush=True)
        if transcript is not None:
            transcript.append({
                "round": r,
                "served_model": data.get("model"),
                "stop_reason": data.get("stop_reason"),
                "usage": usage,
                "blocks": [{
                    "type": b["type"],
                    "head": (b.get("text") or b.get("name") or "")[:400],
                    **({"input": b.get("input") or {}} if b.get("type") == "tool_use" else {}),
                } for b in blocks],
            })
        if tool_uses:
            # execute all tool calls of this round
            results = []
            for tu in tool_uses:
                name = tu.get("name")
                args = tu.get("input") or {}
                try:
                    out = exec_tool_safe(name, args, ctx)
                except Exception as e:
                    out = {"text": f"{name} error: {e}"}
                results.append({"type": "tool_result", "tool_use_id": tu.get("id"),
                                "content": _render_tool_result(out)})
                trail.append({"tool": name, "args": args, "elapsed_ms": int((time.time() - t0) * 1000),
                              "summary": _trim(out.get("text", "")[:220])})
            msgs.append({"role": "assistant", "content": blocks})
            # append tool results as user message with images
            content = []
            for res in results:
                content.append({"type": "tool_result", "tool_use_id": res["tool_use_id"],
                                "content": res["content"]})
            # native tool_result wrapped in user role with text fallback:
            msgs.append({"role": "user", "content": content})
            continue
        parsed = _parse_json(texts)
        if isinstance(parsed, dict) and parsed.get("answer") is not None:
            parsed["tools_used"] = [t["tool"] for t in trail]
            parsed["trail"] = trail
            parsed["n_rounds"] = r + 1
            parsed["texts"] = texts[:1500]
            parsed["requested_model"] = MODEL
            parsed["evaluator_model_label"] = MODEL_LABEL
            parsed["served_models"] = served_models
            parsed["usage_rounds"] = usage_rounds
            parsed["usage_totals"] = _sum_usage([x["usage"] for x in usage_rounds])
            parsed["shared_context_used"] = bool(shared_context)
            return parsed
        # malformed: ask to comply
        msgs.append({"role": "user", "content": [{"type": "text",
                      "text": 'That did not match the answer contract. Reply with only the JSON: '
                              '{"answer": "<one option>", "confidence": "...", "reasoning": "..."}'}]})
    return {"answer": None, "confidence": "low", "reasoning": "max rounds exceeded",
            "tools_used": [t["tool"] for t in trail], "trail": trail,
            "n_rounds": max_rounds, "requested_model": MODEL,
            "evaluator_model_label": MODEL_LABEL, "served_models": served_models,
            "usage_rounds": usage_rounds,
            "usage_totals": _sum_usage([x["usage"] for x in usage_rounds]),
            "shared_context_used": bool(shared_context)}


def exec_tool_safe(name, args, ctx):
    from tools_api import exec_tool
    return exec_tool(name, args, ctx)


def _render_tool_result(out):
    parts = []
    if out.get("text"):
        parts.append({"type": "text", "text": out["text"]})
    if out.get("image"):
        parts.append({"type": "image", "source": {"type": "base64",
                     "media_type": "image/jpeg", "data": out["image"]}})
    return parts if parts else [{"type": "text", "text": "(empty result)"}]


def _trim(s, n=120):
    return s if len(s) <= n else s[:n] + "..."
