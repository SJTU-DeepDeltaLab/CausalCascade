# Diagnostic VQA

Evaluate generated videos on event occurrence, causal pathway, temporal ordering, outcome/termination, and scene consistency.

Run commands from the repository root. Install the core evaluator dependencies first:

```bash
pip install -e ".[vqa]"
```

The four event-based dimensions use a multi-round vision agent. Scene consistency uses SAM2/CV evidence and fixed rules. This automated pipeline scores generated videos; the dataset's reference answers remain the evaluation targets.

**Mode B is reference-assisted:** the agent can inspect both generated and GT videos. Mode A blocks GT video access. Reference answer strings are never supplied to the agent; they are used for final scoring only. Do not mix the two modes in a leaderboard.

Inputs: generated `<series>.<scene>.mp4` videos and the prepared `data/reference` and `data/reference_flat` directories. Manifests bind videos to the canonical `diagnostic_vqa.json` questions.

## Prepare Manifests

```bash
python metrics/vqa/prepare_agentic_model_level_manifest.py \
  --dataset-root data/reference --generated-root outputs/L1/MODEL \
  --ground-truth-root data/reference_flat --model-name MODEL --prompt-level L1 \
  --output-dir work/agentic

python metrics/vqa/prepare_scene_consistency_model_level_manifest.py \
  --dataset-root data/reference --generated-root outputs/L1/MODEL \
  --ground-truth-root data/reference_flat --model-name MODEL --prompt-level L1 \
  --output-dir work/scene_manifest
```

## Independent Votes

Copy `metrics/vqa/.env.example` to `.env` in the same directory. Set the API key, vision model, and Anthropic-compatible endpoint. Calls send video images and may incur charges. Defaults: 32,768 max tokens, 24,576 thinking tokens, temperature 0.2; returned model IDs and usage are logged.

```bash
python metrics/vqa/agent_judge.py \
  --manifest work/agentic/manifest.jsonl --mode B --shared-context \
  --votes 1 --max_rounds 8 --concurrency 3 --tag MODEL_L1_pass1
# Repeat with --tag MODEL_L1_pass2 for an independent second vote.

python metrics/vqa/prepare_adaptive_pass3.py \
  --manifest work/agentic/manifest.jsonl \
  --pass1 metrics/vqa/results/MODEL_L1_pass1/agent_answers.jsonl \
  --pass2 metrics/vqa/results/MODEL_L1_pass2/agent_answers.jsonl \
  --output work/pass3_manifest.jsonl
```

If the third-pass manifest is nonempty, run the agent on it with a new tag, then merge:

```bash
python metrics/vqa/adaptive_vote_merge.py \
  --manifest work/agentic/manifest.jsonl \
  --pass1 metrics/vqa/results/MODEL_L1_pass1/agent_answers.jsonl \
  --pass2 metrics/vqa/results/MODEL_L1_pass2/agent_answers.jsonl \
  --pass3 metrics/vqa/results/MODEL_L1_pass3/agent_answers.jsonl \
  --disagreement-json work/disagreements.json --output work/merged.jsonl
```

Omit `--pass3` when there are no disagreements. Three distinct answers require explicit human adjudication; `apply_manual_adjudications.py` applies those decisions without using GT answers to resolve ties.

## Scene Consistency And Scoring

Install the scene extras and the dependencies required by your chosen SAM2/Transformers version. The following Linux/Bash runner downloads the SAM2 checkpoint if it is not cached and uses CPU execution:

```bash
pip install -e ".[scene]"
bash metrics/vqa/run_scene_consistency_unified_model_level.sh \
  MODEL_L1 work/scene_manifest/manifest.jsonl work/scene_output \
  facebook/sam2.1-hiera-tiny

python metrics/vqa/score_hybrid_manifest.py \
  --dataset-root data/reference \
  --agentic-manifest work/agentic/manifest.jsonl --agentic-answers work/merged.jsonl \
  --scene-manifest work/scene_manifest/manifest.jsonl \
  --scene-predictions work/scene_output/scene_consistency_200_exact_predictions.json \
  --output work/hybrid_score.json
```

For each scene and dimension, average exact-answer accuracy across its questions. Average these scene scores within each dimension, then average the five dimension scores for Overall. Unresolved/missing answers are not silently removed; resolve flagged cases before reporting the final score. Fixed scene-consistency rules are heuristic diagnostics and can miss artifacts, especially under occlusion or tracking errors; validate representative predictions visually.
