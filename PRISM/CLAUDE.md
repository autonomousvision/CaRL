# PRISM — Claude Code Project Brief

## What This Project Is

PRISM (Pareto-optimal Risk-constrained drIving Style Manifold) is a
research codebase implementing a safe, personalised autonomous driving
planner. It learns K=5 policies forming a CVaR-safe Distributional
Pareto Front of driving styles (comfort-seeking, progress-oriented,
disciplined, cautious, balanced) on the nuPlan benchmark.

This is a PhD research project targeting IEEE IV / IROS. The paper
draft is in `docs/prism_paper.tex`. Every implementation decision must
be consistent with the formulation in that paper.

---

## Repository Structure

```
prism/
├── CLAUDE.md                   ← this file (read first, always)
├── hyperparams.json            ← cached hyperparameters (auto-generated)
├── compute_hyperparams.py      ← run once before any training
├── docs/
│   ├── prism_paper.tex         ← full paper draft
│   └── references.bib
├── prism/
│   ├── env/
│   │   ├── nuplan_env.py       ← nuPlan gym wrapper
│   │   ├── rewards.py          ← all four style reward functions
│   │   ├── safety_cost.py      ← two-tier safety cost c_t
│   │   └── regime_detector.py  ← free-flow / car-following / congested
│   ├── morl/
│   │   ├── utility_functions.py ← non-decreasing neural networks (Stage 1)
│   │   ├── dpmorl_trainer.py   ← Stage 2 policy optimisation
│   │   └── cvar_lagrangian.py  ← CVaR estimation + Lagrange update
│   ├── curriculum/
│   │   └── alpha_schedule.py   ← alpha(n) and epsilon(n) curves
│   └── utils/
│       ├── hyperparams.py      ← load hyperparams.json
│       └── zt_normaliser.py    ← z_t running normalisation with EMA
├── scripts/
│   ├── train.py                ← main training entry point
│   ├── evaluate.py             ← evaluation on nuPlan Val14
│   └── visualise_pareto.py     ← plot the distributional Pareto front
├── configs/
│   └── prism_default.yaml      ← all training hyperparameters
└── tests/
    └── test_rewards.py         ← unit tests for reward functions
```

---

## External Repositories and Packages

### Base RL Planner — CaRL
- **Repo**: https://github.com/autonomousvision/CaRL
- **What we use**: observation space (BEV semantic segmentation), nuPlan
  environment wrappers, PPO training infrastructure, scenario filtering.
- **What we do NOT use**: CaRL's reward function (route completion + penalties).
  We replace this entirely with our style reward vector.
- **Key files to reference**: `carl/environment/`, `carl/training/ppo_trainer.py`
- **Do NOT copy CaRL's reward code** into `prism/env/rewards.py`.

### DPMORL — Distributional Pareto MORL
- **Repo**: https://github.com/zpschang/DPMORL
- **Paper**: NeurIPS 2023 — Cai et al., "Distributional Pareto-Optimal
  Multi-Objective Reinforcement Learning"
- **What we use**:
  - Non-decreasing neural network architecture for utility functions
    (`utility_functions.py`)
  - Diversity-based objective for generating K diverse utility functions
  - State augmentation with cumulative returns z_t (Algorithm 1)
  - Scalar reward transformation R_t = gamma^{-t}[f(z_{t+1}) - f(z_t)]
- **Key files**: `dpmorl/utility_function.py`, `dpmorl/algorithm.py`
- **What we add on top**: CVaR Lagrangian constraint (not in DPMORL)

### nuPlan
- **Repo**: https://github.com/motional/nuplan-devkit
- **What we use**: closed-loop simulation, scenario database, IDM planner
  for warm-up rollouts, Val14 evaluation protocol
- **Install**: follow nuplan-devkit README for dataset setup
- **Key classes**:
  - `nuplan.planning.simulation.planner.idm_planner.IDMPlanner`
  - `nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario`
  - nuPlan metrics: `nuplan.planning.metrics`

### Stable-Baselines3
- **Package**: `stable-baselines3`
- **What we use**: PPO implementation as the inner RL algorithm
- **Important**: we use SB3's PPO but with a custom reward that includes
  the Lagrangian penalty. The PPO algorithm itself is unmodified.

---

## Core Mathematical Formulation

Read `docs/prism_paper.tex` for full derivations. Key equations:

### Style Reward Vector (4-dimensional, all in (0,1])

**Comfort** (Eq. comfort):
```
r_comfort = exp(-(j_lon^2 + j_lat^2) / sigma_j_sq)
```

**Progress** (Eq. progress):
```
r_progress = r_speed * r_accel * (0.5 + 0.5 * r_lane)

r_speed = exp(-max(0, v_des - v_ego) / (beta * v_des))
r_accel = 1 - exp(-|a_ego| / gamma_a)
r_lane  = lane_index / (N_lanes - 1)   [0.5 if N_lanes == 1]
```

**Desired speed regime detection** (check in this order):
1. Congested:     v_lane_avg < 0.5 * v_limit  →  v_des = v_lane_avg
2. Car following: lead vehicle in ego lane within horizon  →  v_des = v_lead
3. Free flow:     otherwise  →  v_des = v_limit

**Lateral discipline** (Eq. lateral):
```
r_lateral = r_dev * r_heading

r_dev     = 0.3 + 0.7 * exp(-(d_lat / sigma_d)^2)   [sigma_d = 0.2m]
r_heading = exp(-(|delta_psi| / phi)^2)
```

**Spacing** (Eq. spacing):
```
r_spacing = 0.2 + 0.8 * (1 - exp(-(max(0, TTC) / tau)^2))

TTC = d_lead / (v_ego - v_lead)   when v_ego > v_lead
TTC = inf                          otherwise or no lead vehicle
```

### Safety Cost (two-tier)

```
c_t = c_outcome + sum_j(c_lead_j)
```

**Outcome weights** (from hyperparams.json):
- VRU collision:        100
- Wrong direction:      100
- Vehicle collision:     80
- Red light:             80
- Stop sign:             70
- Drivable area:         65
- Object collision:      40

**Indicator weights** (computed by compute_hyperparams.py):
```
w_j = sum_{i in O_j} W_i / (|O_j| * T_{j->i})
```

**Episode-level cap** for persistent indicators (TTC, THW, speed):
```
sum_t(c_lead_j) <= mean(W_i for i in O_j)
```

### CVaR Constraint with Alpha-Curriculum

```
alpha(n) = 0.20 + 0.75 * min(1, n / N_curriculum)
epsilon(n) = epsilon_curve[alpha(n)]   # from hyperparams.json
CVaR_{alpha(n)}(C^pi_k) <= epsilon(n)
```

### Lagrangian Update

```
effective_reward = R_t - lambda_k * c_t
lambda_k = max(0, lambda_k + eta_lambda * (CVaR_hat - epsilon(n)))
```

### State Augmentation (DPMORL)

```
s_tilde_t = concat(s_t, z_t_normalised)
z_{t+1} = z_t + gamma^t * r_t           # cumulative style returns
z_norm_i = (z_i - mu_i) / sigma_i       # normalised with EMA stats
```

### Scalar Reward (DPMORL Algorithm 1)

```
R_t = gamma^{-t} * [f_theta_k(z_{t+1}) - f_theta_k(z_t)]
```

---

## Hyperparameter File Protocol

**CRITICAL**: `hyperparams.json` must exist before any training.

### To compute (run once):
```bash
python compute_hyperparams.py \
    --output_path hyperparams.json \
    --n_warmup_rollouts 200 \
    --nuplan_data_root /path/to/nuplan \
    --gamma 0.99
```

### In training script — guard pattern:
```python
import subprocess, sys
result = subprocess.run(
    ["python", "compute_hyperparams.py",
     "--check_only", "--output_path", "hyperparams.json"],
    capture_output=True)
if result.returncode != 0:
    print("Hyperparams not found. Run compute_hyperparams.py first.")
    sys.exit(1)
```

### Loading in code:
```python
from prism.utils.hyperparams import load_hyperparams
hp = load_hyperparams("hyperparams.json")
sigma_j_sq = hp["reward_scaling"]["sigma_j_sq"]
epsilon_curve = hp["epsilon_curve"]   # dict: "0.95" -> float
lead_times = hp["lead_times"]         # nested dict
```

---

## Training Entry Point

```bash
python scripts/train.py \
    --config configs/prism_default.yaml \
    --hyperparams hyperparams.json \
    --nuplan_data_root /path/to/nuplan \
    --n_policies 5 \
    --total_timesteps 10000000 \
    --n_rollouts_per_iter 512 \
    --gamma 0.99 \
    --eta_lambda 0.01 \
    --n_curriculum_iters 5000 \
    --output_dir runs/prism_run_001
```

Training runs K=5 policies sequentially (or in parallel if multi-GPU).
Each policy is trained independently with its own lambda_k.

---

## Key Design Decisions (Do Not Change Without Discussion)

1. **No CaRL weight initialisation**: policies train from scratch.
   Rationale: CaRL's scalar reward would bias the style objectives.

2. **IDM warm-up, not random rollouts**: hyperparams computed from IDM
   episodes, not random policy. Rationale: random policy produces
   degenerate trajectories skewed to zero returns.

3. **Episode-level cap for persistent indicators**: TTC, THW, speed
   violation costs are capped per episode. Rationale: prevents
   cumulative indicator cost from exceeding outcome event costs.

4. **Per-policy lambda_k, not shared**: each of K=5 policies has its
   own Lagrange multiplier. Rationale: different style preferences
   produce different safety cost distributions.

5. **Trajectory-level CVaR, not timestep-level**: CVaR is computed
   over C^i = sum(gamma^t * c_t) per rollout. Rationale: captures
   cumulative trajectory danger, not momentary spikes.

6. **Regime check order: congested first**: prevents a slow lead
   vehicle from triggering car-following when the broader lane
   is congested. Always check congestion before car-following.

7. **N=1 lane edge case**: r_lane = 0.5 for single-lane roads.
   Rationale: lane choice is neutral when there is no choice.

---

## Files to Modify When Implementing Each Component

| Component | File |
|---|---|
| Style rewards | `prism/env/rewards.py` |
| Safety cost | `prism/env/safety_cost.py` |
| Regime detection | `prism/env/regime_detector.py` |
| nuPlan gym env | `prism/env/nuplan_env.py` |
| Utility functions | `prism/morl/utility_functions.py` |
| DPMORL Stage 2 | `prism/morl/dpmorl_trainer.py` |
| CVaR + Lagrangian | `prism/morl/cvar_lagrangian.py` |
| Alpha curriculum | `prism/curriculum/alpha_schedule.py` |
| z_t normalisation | `prism/utils/zt_normaliser.py` |
| Hyperparam loading | `prism/utils/hyperparams.py` |
| Hyperparam computation | `compute_hyperparams.py` (root level) |
| Main training loop | `scripts/train.py` |
| Evaluation | `scripts/evaluate.py` |

---

## Common Tasks for Claude Code

### Add a new reward component
1. Implement in `prism/env/rewards.py` following the existing pattern
2. Ensure output is in (0,1]
3. Add corresponding scaling param to `compute_hyperparams.py`
4. Add unit test in `tests/test_rewards.py`
5. Update paper equation in `docs/prism_paper.tex`

### Modify the safety cost
1. Edit `prism/env/safety_cost.py`
2. Update `INDICATOR_OUTCOME_MAP` or `OUTCOME_WEIGHTS` in
   `compute_hyperparams.py` if adding new signals
3. Delete `hyperparams.json` and rerun `compute_hyperparams.py`
4. Update Table I in `docs/prism_paper.tex`

### Change the alpha curriculum
1. Edit `prism/curriculum/alpha_schedule.py`
2. Update alpha_start/alpha_end in `configs/prism_default.yaml`
3. The epsilon curve is always read from `hyperparams.json` --
   no need to rerun hyperparams if only changing alpha_start/end

### Add a new policy (change K)
1. Change `n_policies` in `configs/prism_default.yaml`
2. Stage 1 will automatically generate K utility functions
3. Stage 2 loops over K -- no other changes needed

---

## Important Conventions

- **nuPlan runs at 10 Hz**: all time thresholds in paper (1.5s TTC,
  2.0s THW) correspond to 15 and 20 timesteps respectively.
- **Rewards are always in (0,1]**: never return 0 or negative from
  style reward functions. Use floor values (delta_d=0.3, delta_s=0.2).
- **Safety cost is always >= 0**: never return negative safety cost.
- **gamma = 0.99** everywhere: in rewards, z_t accumulation, and
  CVaR computation. Changing this requires rerunning hyperparams.
- **JSON keys for epsilon_curve are strings**: e.g. "0.95", not 0.95.
  Always use `str(alpha)` as key and `float(hp["epsilon_curve"][str(alpha)])`.
- **EMA beta = 0.01** for z_t normalisation update during training.
- **Blind spot geometry**: defined as ±45° to ±135° from ego heading,
  within 10m longitudinal and 4m lateral of ego centre.

---

## Dependencies

```
nuplan-devkit          # nuPlan simulation and data
stable-baselines3      # PPO implementation
torch                  # neural networks
numpy
scipy
matplotlib             # Pareto front visualisation
wandb                  # experiment tracking (optional)
```

Install:
```bash
pip install stable-baselines3 torch numpy scipy matplotlib
# nuplan-devkit: follow https://github.com/motional/nuplan-devkit
```

---

## Do Not Do

- Do not modify `hyperparams.json` by hand
- Do not add safety signals to the style reward functions
- Do not add style signals to the safety cost
- Do not use a shared lambda across all K policies
- Do not use CaRL's reward function
- Do not hardcode scaling parameters -- always read from hyperparams.json
- Do not compute CVaR at timestep level -- always at trajectory level

---

## Karpathy Agentic Engineering Principles

This project is built under the agentic engineering discipline as defined by Andrej Karpathy (Sequoia Ascent 2026). These are not suggestions — they are the operating philosophy for how this codebase should be written and extended.

### The core distinction

Vibe coding raises the floor — anyone can ship something. Agentic engineering raises the ceiling — professionals ship fast _without sacrificing quality_. This project operates at the ceiling. That means:

- Never blindly accept generated code
- Design specs first, then let agents implement
- Inspect every diff, write tests, create evaluation loops
- Preserve correctness, security, taste, and maintainability throughout

### Software 3.0 thinking

The context window is the primary lever. Before writing any code, ask: what is the right context to give the agent so it can do this correctly? The spec in this `CLAUDE.md` is the Software 3.0 program. The implementation is what emerges from it.

### Macro actions only

Do not work line by line. Work in macro actions:

- "Implement this feature end to end"
- "Refactor this subsystem"
- "Write tests, run them, fix failures"
- "Compare approaches and propose a plan"

If you find yourself making small edits to individual lines, stop and reframe at a higher level.

### The MenuGen mistake — avoid it here

Karpathy's agent matched Stripe emails to Google account emails. Plausible code, broken product logic. The equivalent risk in this project: using session names as IDs instead of UUIDs, or matching users by anything other than `user_id`. Always use persistent, stable identifiers. Never cross-correlate by human-readable fields.

### Verifiability first

Build things that can be verified. Every feature should have a clear pass/fail signal:

- CLI commands should have predictable output that can be asserted
- API calls should return typed responses validated with Zod
- Memory injection should be testable: given X memories, system prompt contains Y
- Session consolidation should be inspectable before saving

### What agents handle vs what humans decide

Agents handle: API syntax, boilerplate, routine CRUD, filling in typed implementations, running and fixing tests.

Humans (you) decide: system boundaries, identity and auth logic, security rules, what makes it into the consolidation, which memories matter, the shape of the data model.

Do not let agents make decisions about: user ID strategy, payment logic, permission scoping, what gets persisted to EverMind.

### Agent-native by design

This product is for agentic engineers. Build it to be used by agents too:

- All CLI commands should be scriptable and produce machine-readable output when passed `--json`
- The `CLAUDE.md` is the agent-native interface to this codebase — keep it current
- Avoid UI-only flows: anything a human can do in the webapp, an agent should be able to do via CLI or API
- Write docs for agents first: precise, copy-pasteable, no "click here" instructions

### Taste matters

The agentic engineer is still in charge of aesthetics, judgment, and taste. Generated code can be bloated, copy-pasted, awkwardly abstracted, and brittle. It is your job to reject that. Specifically:

- No unnecessary abstraction layers — if two things do the same job, pick one
- No copy-pasted blocks — extract to a shared utility
- No magic strings — everything named, typed, and in `constants/`
- No silent failures — every async operation has explicit error handling

### You cannot outsource understanding

The quote that guides this project: _"You can outsource your thinking, but you can't outsource your understanding."_ Know why every architectural decision was made. If you cannot explain it, do not commit it.
