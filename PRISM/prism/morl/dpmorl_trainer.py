"""
DPMORL Stage 2 — Per-policy PPO training with CVaR Lagrangian.

One DPMORLTrainer instance manages a single policy k.  The outer training
loop (scripts/train.py) creates K trainers and calls them sequentially.

Training loop (per policy):
    for update in range(n_updates):
        1. Collect rollouts with current PRISMEnv (scalar reward = R_t - lambda * c_t)
        2. Update PPO policy
        3. Compute cumulative episode costs from rollouts
        4. Update alpha/epsilon via curriculum
        5. Update lambda_k via CVaR Lagrangian
        6. Save checkpoint

Rollout buffer tracks:
    - obs (generic dict — whatever keys PRISMEnv returns; backend-agnostic)
    - actions, log_probs, values, rewards, dones
    - episode_zt (for CVaR computation at episode end)
    - episode_costs (step-level safety costs, for discounted sum)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from prism.curriculum.alpha_schedule import AlphaSchedule
from prism.models.base import PRISMPolicyBase
from prism.morl.cvar_lagrangian import CVaRLagrangian, compute_episode_cost
from prism.morl.utility_functions import UtilityFunction

logger = logging.getLogger(__name__)


@dataclass
class RolloutBuffer:
    """
    Lightweight buffer for one PPO update iteration.

    obs stores each step's full observation dict as a numpy dict,
    preserving whatever keys and dtypes PRISMEnv returns.  This makes
    the buffer backend-agnostic: CaRL obs (bev_semantics + measurements)
    and Alpamayo obs (cameras + ...) are stored identically.
    """
    obs: List[dict] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    # Per-step safety costs for episode CVaR tracking
    step_costs: List[float] = field(default_factory=list)
    # Episode-level z_T vectors for logging
    episode_zts: List[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        for lst in vars(self).values():
            if isinstance(lst, list):
                lst.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class DPMORLTrainer:
    """
    Manages one policy in the PRISM MORL system.

    Parameters
    ----------
    policy_id : int
        Index k ∈ {0, ..., K-1}.
    agent : PRISMPolicyBase
        Policy adapter (CaRLPPOAdapter, AlpamayoAdapter, etc.).
    optimizer : torch.optim.Optimizer
    env : PRISMEnv
        The gymnasium environment.  utility_fn and lambda_k are set here.
    utility_fn : UtilityFunction
        The utility function f_k for this policy.
    hp : Dict
        Loaded hyperparams.json.
    cfg : dict
        Training config (from prism_default.yaml).
    output_dir : Path
        Where to save checkpoints.
    device : torch.device
    """

    def __init__(
        self,
        policy_id: int,
        agent: PRISMPolicyBase,
        optimizer: torch.optim.Optimizer,
        env,                          # PRISMEnv
        utility_fn: UtilityFunction,
        hp: Dict,
        cfg: dict,
        output_dir: Path,
        device: torch.device,
    ) -> None:
        self.policy_id = policy_id
        self.agent = agent
        self.optimizer = optimizer
        self.env = env
        self.utility_fn = utility_fn
        self.hp = hp
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.device = device

        self.alpha_schedule = AlphaSchedule(
            hp=hp,
            alpha_start=cfg.get("alpha_start", 0.20),
            alpha_end=cfg.get("alpha_end", 0.95),
            n_curriculum=cfg.get("n_curriculum_iters", 5000),
        )
        self.lagrangian = CVaRLagrangian(
            eta_lambda=cfg.get("eta_lambda", 0.01),
            buffer_size=cfg.get("cvar_buffer_size", 500),
        )

        self._global_step = 0
        self._buffer = RolloutBuffer()

        # Set env references
        self.env.set_utility_fn(utility_fn.as_callable())
        self.env.set_lambda(0.0)

    def train(self, n_updates: int, steps_per_update: int = 512) -> Dict:
        """
        Run the full PPO training loop for this policy.

        Returns a summary dict with final lambda, CVaR, and episode z_Ts.
        """
        self.agent.train()
        summary = {
            "policy_id": self.policy_id,
            "lambda_history": [],
            "cvar_history": [],
            "episode_zts": [],
        }

        obs, _ = self.env.reset()
        next_obs = {k: torch.from_numpy(np.array(v)).unsqueeze(0).to(self.device)
                    for k, v in obs.items()}

        for update in range(n_updates):
            alpha, epsilon = self.alpha_schedule.get(update)

            # ── Rollout collection ─────────────────────────────────────
            self._buffer.clear()
            episode_step_costs: List[float] = []

            for _ in range(steps_per_update):
                self._global_step += 1

                with torch.no_grad():
                    out = self.agent.forward(next_obs)

                act_np = out.action.squeeze(0).cpu().numpy()
                next_obs_raw, reward, term, trunc, info = self.env.step(act_np)
                done = term or trunc

                c_t = float(info.get("safety_cost", 0.0))
                episode_step_costs.append(c_t)

                self._buffer.obs.append({
                    k: v.squeeze(0).cpu().numpy()
                    for k, v in next_obs.items()
                })
                self._buffer.actions.append(act_np)
                self._buffer.log_probs.append(float(out.log_prob.item()))
                self._buffer.values.append(float(out.value.item()))
                self._buffer.rewards.append(float(reward))
                self._buffer.dones.append(done)
                self._buffer.step_costs.append(c_t)

                if done:
                    # Record episode-level cumulative cost for CVaR
                    ep_cost = compute_episode_cost(
                        episode_step_costs, gamma=self.cfg.get("gamma", 0.99)
                    )
                    self.lagrangian.add_episode(ep_cost)
                    episode_step_costs = []

                    if "episode_zt" in info:
                        self._buffer.episode_zts.append(info["episode_zt"].copy())
                        summary["episode_zts"].append(info["episode_zt"].copy())

                    obs, _ = self.env.reset()
                    next_obs = {
                        k: torch.from_numpy(np.array(v)).unsqueeze(0).to(self.device)
                        for k, v in obs.items()
                    }
                else:
                    next_obs = {
                        k: torch.from_numpy(np.array(v)).unsqueeze(0).to(self.device)
                        for k, v in next_obs_raw.items()
                    }

            # ── PPO update ─────────────────────────────────────────────
            self._ppo_update(next_obs)

            # ── Lagrangian update ──────────────────────────────────────
            new_lambda, cvar_hat = self.lagrangian.update_lambda(alpha, epsilon)
            self.env.set_lambda(new_lambda)

            summary["lambda_history"].append(new_lambda)
            summary["cvar_history"].append(cvar_hat)

            if (update + 1) % 100 == 0:
                logger.info(
                    f"[Policy {self.policy_id}] update={update+1}/{n_updates}  "
                    f"alpha={alpha:.2f}  epsilon={epsilon:.3f}  "
                    f"CVaR={cvar_hat:.3f}  lambda={new_lambda:.4f}"
                )

            if (update + 1) % self.cfg.get("save_every", 500) == 0:
                self._save_checkpoint(update)

        self._save_checkpoint(n_updates - 1, final=True)
        return summary

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, last_obs: dict) -> None:
        """Single PPO update using the collected rollout buffer."""
        gamma = self.cfg.get("gamma", 0.99)
        gae_lambda = self.cfg.get("gae_lambda", 0.95)
        clip_coef = self.cfg.get("clip_coef", 0.2)
        vf_coef = self.cfg.get("vf_coef", 0.5)
        ent_coef = self.cfg.get("ent_coef", 0.01)
        update_epochs = self.cfg.get("update_epochs", 4)
        minibatch_size = self.cfg.get("minibatch_size", 64)
        max_grad_norm = self.cfg.get("max_grad_norm", 0.5)

        n = len(self._buffer)
        if n == 0:
            return

        rewards = np.array(self._buffer.rewards, dtype=np.float32)
        values = np.array(self._buffer.values, dtype=np.float32)
        dones = np.array(self._buffer.dones, dtype=np.float32)
        log_probs = np.array(self._buffer.log_probs, dtype=np.float32)
        actions = np.array(self._buffer.actions, dtype=np.float32)

        # Bootstrap value
        with torch.no_grad():
            next_value = float(self.agent.forward(last_obs).value.item())

        # GAE advantages
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else values[t + 1]
            non_term = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * non_term - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * non_term * last_gae
        returns = advantages + values

        # Convert to tensors — obs keys and dtypes are preserved from the buffer
        b_obs = {
            k: torch.from_numpy(
                np.stack([step[k] for step in self._buffer.obs])
            ).to(self.device)
            for k in self._buffer.obs[0].keys()
        }
        b_actions = torch.from_numpy(actions).to(self.device)
        b_log_probs = torch.from_numpy(log_probs).to(self.device)
        b_advantages = torch.from_numpy(advantages).to(self.device)
        b_returns = torch.from_numpy(returns).to(self.device)
        b_values = torch.from_numpy(values).to(self.device)

        for _ in range(update_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, minibatch_size):
                mb = perm[start : start + minibatch_size]
                mb_obs = {k: v[mb] for k, v in b_obs.items()}
                mb_out = self.agent.forward(mb_obs, actions=b_actions[mb])

                new_value = mb_out.value.view(-1)
                log_ratio = mb_out.log_prob - b_log_probs[mb]
                ratio = log_ratio.exp()

                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                ).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                ent_loss = mb_out.entropy.mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.trainable_parameters()), max_grad_norm
                )
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, update: int, final: bool = False) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tag = "final" if final else f"{update:09d}"
        model_path = self.output_dir / f"policy_{self.policy_id}_model_{tag}.pth"
        util_path = self.output_dir / f"policy_{self.policy_id}_utility_{tag}.pth"
        lagr_path = self.output_dir / f"policy_{self.policy_id}_lagrangian_{tag}.json"

        torch.save(self.agent.state_dict(), model_path)
        torch.save(self.utility_fn.state_dict(), util_path)

        import json
        with open(lagr_path, "w") as f:
            json.dump(self.lagrangian.state_dict(), f)

        logger.info(f"[Policy {self.policy_id}] Saved checkpoint: {tag}")
