# PRISM — Training Guide

Pareto-optimal Risk-constrained drIving Style Manifold.  
Two backbone options: **CaRL PPO** (BEV raster) and **Alpamayo** (camera + VLM).

---

## Prerequisites

### 1. Install dependencies

**Both backbones:**
```bash
pip install stable-baselines3 torch numpy scipy matplotlib opencv-python gymnasium
# nuplan-devkit: follow https://github.com/motional/nuplan-devkit
```

**Alpamayo only:**
```bash
pip install transformers          # backbone loading
pip install peft                  # Phase B LoRA fine-tuning only
```

### 2. Set data paths

```bash
export NUPLAN_DATA_ROOT=/data/nuplan/dataset
export NUPLAN_MAP_ROOT=/data/nuplan/maps
```

**Alpamayo only** — sensor blob root for JPEG camera images:
```bash
export NUPLAN_SENSOR_ROOT=/data/nuplan/sensor_blobs
```
If `NUPLAN_SENSOR_ROOT` is not set, the camera observation builder returns zero
frames and logs a warning. Training proceeds structurally but without real images.

### 3. Build the scenario cache

Run once before any training:
```bash
cd nuPlan && python carl_nuplan/planning/script/run_gym.py \
    py_func=cache \
    cache.cache_path=/data/prism_cache \
    +scenario_builder=gym_nuplan \
    +scenario_filter=train150k_split \
    scenario_builder.data_root=$NUPLAN_DATA_ROOT \
    scenario_builder.map_root=$NUPLAN_MAP_ROOT
```

Update `cache_path` in your config file to match.

### 4. Compute hyperparameters

Run once before any training. Requires the nuPlan dataset to be accessible.

```bash
python compute_hyperparams.py \
    --output_path hyperparams.json \
    --n_warmup_rollouts 200 \
    --nuplan_data_root $NUPLAN_DATA_ROOT \
    --gamma 0.99
```

All training commands below guard against a missing `hyperparams.json` and will
exit with a clear error if this step was skipped.

---

## Training — CaRL PPO backbone

Uses CaRL's BEV semantic segmentation observation space.

```bash
python scripts/train.py \
    --config configs/prism_default.yaml \
    --hyperparams hyperparams.json \
    --nuplan_data_root $NUPLAN_DATA_ROOT \
    --n_policies 5 \
    --total_timesteps 10000000 \
    --output_dir runs/carl_run_001
```

No backbone phase flags needed. A single training run covers both Stage 1
(utility function initialisation) and Stage 2 (per-policy PPO with CVaR
Lagrangian) for all K=5 policies sequentially.

---

## Training — Alpamayo backbone

Alpamayo uses raw camera images (CAM_F0, CAM_L0, CAM_R0) and a 10B VLM
backbone. Training is split into two phases to protect the pretrained
representations.

### Phase A — frozen backbone

Only the stochastic action head and Q-Former critic train. Run this first.

```bash
python scripts/train.py \
    --config configs/prism_alpamayo.yaml \
    --hyperparams hyperparams.json \
    --nuplan_data_root $NUPLAN_DATA_ROOT \
    --backbone-phase a \
    --total_timesteps 5000000 \
    --output_dir runs/alpamayo_phase_a
```

The backbone (`nvidia/Alpamayo-R1-10B`) is downloaded from HuggingFace on first
run. To use a local copy, set `backbone_model_name` in
`configs/prism_alpamayo.yaml` to the local path.

### Phase B — LoRA fine-tuning

Requires a completed Phase A output directory. LoRA adapters are added to the
VLM (rank 16) and action expert (rank 32). The CVaR buffer resets so it
reflects only Phase B experience.

```bash
python scripts/train.py \
    --config configs/prism_alpamayo.yaml \
    --hyperparams hyperparams.json \
    --nuplan_data_root $NUPLAN_DATA_ROOT \
    --backbone-phase b \
    --phase-a-dir runs/alpamayo_phase_a \
    --total_timesteps 2000000 \
    --output_dir runs/alpamayo_phase_b
```

---

## Switching backbones at the command line

The `--model-backend` flag overrides the `model_backend` field in the config
without editing any file:

```bash
# Force CaRL PPO regardless of what the config says
python scripts/train.py --config configs/prism_alpamayo.yaml \
    --model-backend carl_ppo ...

# Force Alpamayo
python scripts/train.py --config configs/prism_default.yaml \
    --model-backend alpamayo ...
```

> **Note:** backend-specific config sections (`alpamayo_obs`, `alpamayo.*`)
> must still be present in the YAML for the chosen backend. Using
> `configs/prism_alpamayo.yaml` as the base config is safest.

---

## Evaluation

Evaluate a trained run on the nuPlan Val14 benchmark:

```bash
python scripts/evaluate.py \
    --policy_dir runs/alpamayo_phase_b \
    --config configs/prism_alpamayo.yaml \
    --hyperparams hyperparams.json \
    --n_episodes 200 \
    --output_dir runs/alpamayo_phase_b/eval
```

---

## Stage 1 utilities — advanced options

Run Stage 1 only and save utility functions for reuse:
```bash
python scripts/train.py --config configs/prism_default.yaml \
    --hyperparams hyperparams.json \
    --stage1_only \
    --output_dir runs/stage1_only
```

Skip Stage 1 and load pre-saved utility functions:
```bash
python scripts/train.py --config configs/prism_default.yaml \
    --hyperparams hyperparams.json \
    --skip_stage1 \
    --utility_fn_dir runs/stage1_only/stage1 \
    --output_dir runs/carl_run_002
```

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/prism_default.yaml` | YAML config file |
| `--hyperparams` | `hyperparams.json` | Pre-computed hyperparameter file |
| `--nuplan_data_root` | — | Path to nuPlan dataset root |
| `--n_policies` | from config | Number of policies K |
| `--total_timesteps` | from config | Total env steps across all policies |
| `--output_dir` | from config | Directory for checkpoints and summaries |
| `--seed` | from config | Random seed |
| `--model-backend` | from config | `carl_ppo` or `alpamayo` — overrides config |
| `--backbone-phase` | — | `a` (frozen) or `b` (LoRA) — Alpamayo only |
| `--phase-a-dir` | — | Phase A output dir — required for `--backbone-phase b` |
| `--stage1_only` | false | Run Stage 1 and exit |
| `--skip_stage1` | false | Skip Stage 1, load from `--utility_fn_dir` |
| `--utility_fn_dir` | — | Directory containing saved utility function checkpoints |

---

## Output structure

```
runs/
└── alpamayo_phase_a/
    ├── stage1/
    │   ├── utility_fn_0.pth … utility_fn_4.pth
    └── policy_{k}/
        ├── policy_{k}_model_final.pth
        ├── policy_{k}_utility_final.pth
        ├── policy_{k}_lagrangian_final.json
        └── training_summary.json
```

Phase B loads `policy_{k}_model_final.pth` from the Phase A directory via
`--phase-a-dir`. The Lagrangian state is always reset fresh for Phase B.
