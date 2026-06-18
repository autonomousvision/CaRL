# DPMORL with CVaR-Constrained Safety in CaRL

Integration of the Distributional Pareto-Optimal Multi-Objective Reinforcement Learning (DPMORL) approach with CaRL (nuPlan simulation framework), adding a Conditional Value at Risk (CVaR) safety constraint to PPO.

## Research Project Feedback & Analysis

This research project is highly promising, timely, and addresses key limitations in both Multi-Objective RL and autonomous driving safety. 

### Why this is a strong design:
1. **Multi-Objective vs. Single Scalar Reward**: Standard CaRL aggregates all objectives (progress, comfort, lane-keeping, and safety) into a single scalar reward using multiplication or addition. This assumes a fixed trade-off. In the real world, passengers have different preferences (e.g., a "sporty" passenger prefers faster progress even if jerk is slightly higher; a "defensive" passenger prioritizes comfort and distance from other vehicles). MORL naturally handles this.
2. **Distributional MORL (DPMORL)**: Real-world driving contains significant stochasticity (e.g., pedestrian behavior, traffic light timings). Optimizing expected values (as in standard MORL) can result in policies that are unsafe in the worst-case tail. Modeling the full return distribution via DPMORL helps the policy manage risk effectively across different preferences.
3. **CVaR Safety Constraint**: Safety in autonomous driving is a **non-negotiable constraint**, not a soft trade-off. Standard MORL might trade collision risk for higher progress if the utility function allows it. 
   - **Conditional Value at Risk ($\text{CVaR}_\alpha$)** is the expected return in the worst $\alpha$ fraction of outcomes (e.g., the worst 5% of runs). 
   - Restricting or maximizing $\text{CVaR}_\alpha(R_{\text{safety}}) \ge C$ ensures the policy is extremely robust against worst-case edge cases (e.g., rare near-collisions).

### Proposed Mathematical Formulation
We split the objectives into:
*   **Safety Returns ($R^s$)**: Negative of collisions, red lights, off-road violations, and extremely low TTC.
*   **Driving Style Returns ($R^d$)**: Progress, comfort, and lane centering.

We formulate the learning problem as:
$$\max_{\pi} \mathbb{E}[U(z^d)] \quad \text{subject to} \quad \text{CVaR}_\alpha(R^s) \ge C_{\text{safe}}$$
where:
*   $U(z^d)$ is a non-linear utility function trained via DPMORL representing driving style preferences.
*   $z^d$ is the accumulated style reward vector over the trajectory.
*   $\text{CVaR}_\alpha(R^s)$ is the expected safety return under the worst $\alpha$ (e.g., 5%) outcomes.
*   $C_{\text{safe}}$ is a threshold ensuring safety constraints are satisfied.

---

## User Review Required

> [!IMPORTANT]
> **Safety Return Representation**: We need to decide how to represent $R^s$. Typically, $R^s \le 0$ where $0$ means a perfectly safe run, and negative values denote constraint infractions (e.g., collisions, red lights).
> **CVaR Threshold Estimation**: Since CVaR is computed over the worst-case runs, we can estimate it empirically over each batch of trajectories during PPO rollouts, or train a distributional critic (using Quantile Regression) to estimate safety value distributions.

---

## Open Questions

> [!WARNING]
> 1. Do we want to treat safety as a hard constraint solved via Lagrange multipliers (e.g., penalizing the policy with $\beta (C_{\text{safe}} - \text{CVaR}_\alpha(R^s))$ where $\beta$ is updated dynamically), or as a soft penalty in the utility function?
> 2. How many distinct non-safety objectives should we include? We propose 3: (1) Route Progress, (2) Kinematic Comfort, (3) Lane-Keeping Centering.

---

## Proposed Changes

We will create a new directory `carl_safety_morl` in the workspace to contain our implementation.

### [NEW] carl_safety_morl

#### [NEW] [__init__.py](file:///Users/maitrayeekeskar/Documents/Git/MI3/PRISM-for-NuPlan/carl_safety_morl/__init__.py)
Package initializer.

#### [NEW] [cvar_ppo.py](file:///Users/maitrayeekeskar/Documents/Git/MI3/PRISM-for-NuPlan/carl_safety_morl/cvar_ppo.py)
Custom PPO implementation incorporating CVaR safety constraints on the safety reward buffer.

#### [NEW] [vector_reward_builder.py](file:///Users/maitrayeekeskar/Documents/Git/MI3/PRISM-for-NuPlan/carl_safety_morl/vector_reward_builder.py)
A custom reward builder that returns vectorized rewards (separating safety components from style/comfort/progress components).

#### [NEW] [utility_wrapper.py](file:///Users/maitrayeekeskar/Documents/Git/MI3/PRISM-for-NuPlan/carl_safety_morl/utility_wrapper.py)
Env wrapper similar to DPMORL's `MultiEnv_UtilityFunction` adapted for the CaRL gym environment.

#### [NEW] [train.py](file:///Users/maitrayeekeskar/Documents/Git/MI3/PRISM-for-NuPlan/carl_safety_morl/train.py)
Training entry point configuring nuPlan scenarios, observation builder, vectorized reward builder, utility functions, and running the CVaR-constrained PPO agent.

---

## Verification Plan

### Automated/Simulation Tests
- We will write unit tests to verify:
  1. Vectorized reward outputs from `vector_reward_builder.py`.
  2. CVaR calculation logic on a simulated distribution of safety scores.
  3. Utility transformation logic in `utility_wrapper.py`.
- Run single-scenario training tests to verify that the PPO agent executes steps and updates its policy weights without exceptions.
