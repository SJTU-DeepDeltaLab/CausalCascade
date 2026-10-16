# Usage

Run commands from the repository root. For dataset examples and results, see the [README](README.md). For automated diagnostic VQA, see [VQA.md](VQA.md).

## Repository Layout

```text
CausalCascade/
├── causalcascade/       Data preparation and lazy HF dataset loader
├── metrics/
│   ├── motion/          Factual motion and temporal alignment
│   ├── counterfactual/  Paired Effect/Preserve metrics
│   └── vqa/             Video-QA evidence, voting, and scoring
├── examples/
│   └── inference/       Generation adapters for nine I2V models
├── pyproject.toml       Package and dependency configuration
├── README.md
├── USAGE.md             Data preparation, generation, and evaluation
├── VQA.md               Automated VQA setup and commands
└── LICENSE
```

## Installation And Data

Use Python 3.10+. The scene-consistency pipeline requires Linux/Bash; generation dependencies are model-specific.

```bash
pip install -e ".[data,metrics,video]"
python -m causalcascade.prepare_dataset --output-dir data
```

The dataset contains 50 base scenes, 2,832 counterfactual variants, 225 matched factual-counterfactual pairs, and 872 diagnostic questions. Preparation downloads the base references and the 225 pairs, with state-only (L1), trigger-aware (L2), and full event-chain (L3) generation inputs.

## Evaluate Generated Videos

Run commands from the repository root. Generate videos with your own I2V pipeline or the adapters in `examples/inference/`.

### Factual Motion

Name videos `<series>.<scene>.mp4`.

```bash
python metrics/motion/evaluate_video_physics_metrics.py \
  --dataset-root data/reference --generated-dir outputs/L1/MODEL \
  --output-dir outputs/motion/L1/MODEL

# After evaluating L1, L2, and L3 for each model:
python metrics/motion/aggregate_motion.py outputs/motion > outputs/motion_summary.csv
```

This computes Spatial IoU, Weighted Spatial IoU, and Strict ST-IoU, with affine/DTW timing diagnostics. Repeat for L2 and L3. Scores are averaged across scenes.

### Counterfactual *Effect* And *Preserve*

Place factual videos under `outputs/L1/<model>/` and counterfactual videos under `outputs/RQ2/<model>/`. Name edited videos `<edit_type>.<series>.<scene>.<edit>.mp4`; configure model directories in `metrics/counterfactual/model_registry.json`.

```bash
python metrics/counterfactual/compute_cf_motion_metrics.py \
  --gt-root data/reference --gt-cf-root data/counterfactual \
  --factual-root outputs/L1 --counterfactual-root outputs/RQ2 \
  --output-dir outputs/counterfactual --dry-run
```

Check coverage, then repeat without `--dry-run`. The four metrics are Spatial/Weighted Effect IoU and Spatial/Weighted Preserve IoU. Primary scores average edits within each scene, then average scenes separately for *delete*, *move*, *scale*, and *mass*.

### Diagnostic VQA

Follow [VQA.md](VQA.md) to prepare manifests, collect independent answers and scene-consistency evidence, and score the five diagnostic dimensions. API evaluation requires credentials and sends video images to the configured provider.

## Outputs

| Evaluation | Output |
|---|---|
| Factual motion | Per-video reports; aggregated `motion_summary.csv` |
| Counterfactual | `all_case_results.csv`, `leaderboard_scene_macro.csv`, `pairing_manifest.json` |
| Diagnostic VQA | Answer records, visual evidence, and `hybrid_score.json` |

Motion scores average over scenes. VQA first averages questions within each scene and dimension, then scenes, and finally the five dimensions for Overall. Counterfactual scores average edits within each scene, then scenes, separately by edit type; incomplete models are reported separately. Validate completed CF results with:

```bash
python metrics/counterfactual/validate_cf_motion_metrics.py --results-dir outputs/counterfactual
```

## Metrics

| Metric | Measures |
|---|---|
| Spatial IoU | Regions exhibiting motion anywhere in the rollout. |
| Weighted Spatial IoU | Motion activity weighted by each pixel's active-frame fraction. |
| Strict ST-IoU | Motion agreement on the reference physical-time grid. |
| CF Spatial/Weighted Effect IoU | Correctly suppressed and induced motion, preserving change direction. |
| CF Spatial/Weighted Preserve IoU | Motion shared by the factual and counterfactual references. |
| Diagnostic VQA | Event occurrence, causal pathway, temporal ordering, outcome/termination, and scene consistency. |

For factual map `f` and edited map `cf`, define `P = min(f, cf)`, `S = max(f - cf, 0)`, and `A = max(cf - f, 0)`. Effect combines the intersections/unions of the two directional channels `S` and `A`; Preserve compares `P`. Weighted scores use pixelwise min/max. Empty unions score 1. Weighted motion measures temporal occupancy, not speed or event order.

Motion extraction defaults: 256 x 256 grid, threshold 0.06, background rate 0.03, Gaussian kernel 5, morphology kernel 3. Temporal scoring holds the final generated frame if a video ends early; affine/DTW scores are timing diagnostics, not replacements for Strict ST-IoU.

## Video Generation

Adapters in `examples/inference/` read `first_frame.png` and `prompt.txt` and write MP4s plus runtime metadata. Install the selected model's upstream environment and weights separately; evaluation does not require these adapters. Each script accepts `--help`.

```bash
torchrun --nproc_per_node=4 examples/inference/run_wan2_2_i2v_a14b_official_batch.py \
  --dataset_dir data/inputs/L1 --output_dir outputs/L1/Wan2.2-I2V-A14B \
  --model_dir /path/to/Wan2.2-I2V-A14B --code_dir /path/to/Wan2.2 \
  --size '832*480' --frame_num 121 --seed 42 --seed_strategy case_hash \
  --dit_fsdp --t5_fsdp --ulysses_size 4
```

Use `data/inputs/L2` or `L3` for richer future-event specification, or `data/counterfactual` with `outputs/RQ2/<model>` for the paired experiments. Counterfactuals use supplied matched L1 prompts: only intervention-affected attributes change, without reference future events. Delete/move/scale alter the initial image; mass leaves it unchanged. `case_hash` shares a scene-based seed between factual and edited runs.

<details>
<summary>Adapters and reported rollout settings (not necessarily defaults)</summary>

| Model | Script in `examples/inference/` | Resolution / frames / fps |
|---|---|---|
| CogVideoX-Fun-V1.5 (5B InP) | `run_cogvideox_fun_inp_intervention.py` | 1360 x 768 / 85 / 8 |
| Cosmos3-Nano | `run_cosmos3_nano_i2v_diffusers_batch.py` | 832 x 480 / 189 / 24 |
| HunyuanVideo-I2V | `run_hunyuanvideo_i2v_official_batch.py` | 540p / 129 / 24 |
| LTX-Video-13B | `run_ltx_video_13b_intervention.py` | 832 x 480 / 121 / 16 |
| MAGI-1-4.5B | `run_magi1_4p5b_intervention.py` | 832 x 480 / 120 / 24 |
| SkyReels-V2-I2V | `run_skyreels_v2_i2v_14b_720p_intervention.py` | 720p / 121 / 24 |
| Wan2.1-I2V-14B | `run_wan2_1_i2v_14b_480p_direct_batch.py` | 832 x 480 / 121 / 16 |
| Wan2.2-TI2V-5B | `run_wan2_2_ti2v_5b_intervention.py` | 832 x 480 / 121 / 24 |
| Wan2.2-I2V-A14B | `run_wan2_2_i2v_a14b_official_batch.py` | 832 x 480 / 121 / 16 |

Set rollout parameters explicitly where exposed; Wan output fps follows its upstream configuration. Exact historical upstream commits and full environment lockfiles were not archived, so bitwise reproduction is not guaranteed.

</details>
