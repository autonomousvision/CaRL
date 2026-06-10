"""
Unit tests for PRISM style reward functions.

All tests are pure Python / NumPy — no nuPlan or gym dependencies required.
Run with:
    cd PRISM && python -m pytest tests/test_rewards.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.env.rewards import (
    compute_comfort,
    compute_lateral_discipline,
    compute_progress,
    compute_spacing,
    compute_style_rewards,
    compute_ttc,
)
from prism.utils.zt_normaliser import ZtNormaliser
from prism.curriculum.alpha_schedule import AlphaSchedule, compute_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_hp():
    return {
        "reward_scaling": {
            "sigma_j_sq": 1.0,
            "beta": 0.5,
            "gamma_a": 1.0,
            "phi": 0.3,
            "tau": 2.0,
        },
        "floor_values": {"delta_d": 0.2},
        "epsilon_curve": {
            str(round(a, 2)): 50.0 for a in np.arange(0.20, 1.01, 0.05)
        },
        "alpha_curriculum": {"alpha_start": 0.20, "alpha_end": 0.95},
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_comfort
# ─────────────────────────────────────────────────────────────────────────────

class TestComfort:

    def test_zero_jerk_returns_one(self):
        r = compute_comfort(j_lon=0.0, j_lat=0.0, sigma_j_sq=1.0)
        assert r == pytest.approx(1.0, abs=1e-8)

    def test_positive_jerk_less_than_one(self):
        r = compute_comfort(j_lon=1.0, j_lat=0.0, sigma_j_sq=1.0)
        assert 0.0 < r < 1.0

    def test_output_in_zero_one_interval(self):
        for j_lon in np.linspace(-5, 5, 11):
            for j_lat in np.linspace(-5, 5, 11):
                r = compute_comfort(j_lon, j_lat, sigma_j_sq=1.0)
                assert 0.0 < r <= 1.0, f"Out of (0,1] for j_lon={j_lon}, j_lat={j_lat}"

    def test_large_jerk_near_zero_but_positive(self):
        # j=10 m/s³ is already severe (braking hard); gives exp(-200) which is
        # non-zero in float64 (underflow only occurs around j≈95 with sigma=1).
        r = compute_comfort(j_lon=10.0, j_lat=10.0, sigma_j_sq=1.0)
        assert r > 0.0

    def test_lat_lon_symmetric(self):
        r1 = compute_comfort(j_lon=2.0, j_lat=0.0, sigma_j_sq=1.0)
        r2 = compute_comfort(j_lon=0.0, j_lat=2.0, sigma_j_sq=1.0)
        assert r1 == pytest.approx(r2, abs=1e-8)

    def test_sigma_scaling(self):
        # Larger sigma_j_sq → less penalisation → higher reward for same jerk
        r_small = compute_comfort(1.0, 1.0, sigma_j_sq=0.5)
        r_large = compute_comfort(1.0, 1.0, sigma_j_sq=2.0)
        assert r_large > r_small


# ─────────────────────────────────────────────────────────────────────────────
# compute_progress
# ─────────────────────────────────────────────────────────────────────────────

class TestProgress:

    def test_output_in_zero_one_interval(self):
        for v_ego in [0, 5, 10, 15]:
            for v_des in [5, 10, 15]:
                for a_ego in [-2, 0, 2]:
                    r = compute_progress(
                        v_ego=v_ego, v_des=v_des, a_ego=a_ego,
                        lane_index=0, n_lanes=1,
                        beta=0.5, gamma_a=1.0,
                    )
                    assert 0.0 < r <= 1.0, (
                        f"Out of (0,1] for v_ego={v_ego}, v_des={v_des}, a_ego={a_ego}"
                    )

    def test_single_lane_neutral_r_lane(self):
        # Single lane: r_lane = 0.5
        r1 = compute_progress(10, 10, 2, lane_index=0, n_lanes=1, beta=0.5, gamma_a=1.0)
        # Two lanes, ego in middle: r_lane = lane_index / (n_lanes - 1)
        r2_left  = compute_progress(10, 10, 2, lane_index=0, n_lanes=2, beta=0.5, gamma_a=1.0)
        r2_right = compute_progress(10, 10, 2, lane_index=1, n_lanes=2, beta=0.5, gamma_a=1.0)
        # r_lane=0.5 for n_lanes=1, r_lane=0 and 1 for n_lanes=2
        assert r2_left < r1 < r2_right

    def test_at_desired_speed_no_speed_penalty(self):
        # When v_ego == v_des, shortfall = 0, r_speed = 1
        r = compute_progress(10, 10, a_ego=1.0, lane_index=0, n_lanes=1, beta=0.5, gamma_a=1.0)
        # r_speed should be 1.0, r_accel depends on a_ego
        r_speed = 1.0
        r_accel = 1 - math.exp(-1.0 / 1.0)
        r_lane = 0.5
        expected = r_speed * r_accel * (0.5 + 0.5 * r_lane)
        assert r == pytest.approx(expected, rel=1e-6)

    def test_zero_acceleration_reduces_progress(self):
        r_accel_zero = compute_progress(10, 10, 0.0, 0, 1, 0.5, 1.0)
        r_accel_pos  = compute_progress(10, 10, 2.0, 0, 1, 0.5, 1.0)
        # r_accel = 1 - exp(-|a|/gamma_a); zero a → 0, positive a → > 0
        assert r_accel_zero < r_accel_pos


# ─────────────────────────────────────────────────────────────────────────────
# compute_lateral_discipline
# ─────────────────────────────────────────────────────────────────────────────

class TestLateral:

    def test_perfect_alignment_max_reward(self):
        r = compute_lateral_discipline(d_lat=0.0, delta_psi=0.0, sigma_d=0.2, phi=0.3)
        expected = (0.3 + 0.7 * 1.0) * 1.0
        assert r == pytest.approx(expected, abs=1e-6)

    def test_output_in_zero_one_interval(self):
        for d_lat in [0, 0.1, 0.5, 1.0, 2.0]:
            for delta_psi in [0, 0.1, 0.5, 1.0, math.pi]:
                r = compute_lateral_discipline(d_lat, delta_psi, sigma_d=0.2, phi=0.3)
                assert 0.0 < r <= 1.0

    def test_floor_value(self):
        # r_dev has floor of 0.3 (when d_lat -> inf)
        r = compute_lateral_discipline(d_lat=1000.0, delta_psi=0.0, sigma_d=0.2, phi=0.3)
        assert r >= 0.3 - 1e-6  # r_dev >= 0.3, r_heading = 1.0

    def test_large_heading_error(self):
        r_small = compute_lateral_discipline(0.0, 0.1, 0.2, 0.3)
        r_large = compute_lateral_discipline(0.0, 1.5, 0.2, 0.3)
        assert r_large < r_small


# ─────────────────────────────────────────────────────────────────────────────
# compute_spacing / compute_ttc
# ─────────────────────────────────────────────────────────────────────────────

class TestSpacing:

    def test_no_lead_vehicle_max_spacing(self):
        ttc = compute_ttc(d_lead=50.0, v_ego=10.0, v_lead=0.0, has_lead=False)
        assert ttc == float("inf")
        r = compute_spacing(ttc, tau=2.0)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_closing_fast_low_spacing(self):
        ttc = compute_ttc(d_lead=10.0, v_ego=20.0, v_lead=0.0, has_lead=True)
        assert ttc == pytest.approx(0.5, rel=1e-6)
        r = compute_spacing(ttc, tau=2.0)
        assert r < 0.5  # low reward when TTC is very small

    def test_output_in_zero_one_interval(self):
        for ttc in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]:
            r = compute_spacing(ttc, tau=2.0)
            assert 0.0 < r <= 1.0, f"Out of (0,1] for ttc={ttc}"

    def test_floor_value(self):
        r = compute_spacing(ttc=0.0, tau=2.0)
        assert r == pytest.approx(0.2, abs=1e-6)  # floor = 0.2

    def test_not_closing_returns_inf_ttc(self):
        ttc = compute_ttc(d_lead=20.0, v_ego=5.0, v_lead=10.0, has_lead=True)
        assert ttc == float("inf")  # not closing (v_lead > v_ego)


# ─────────────────────────────────────────────────────────────────────────────
# compute_style_rewards (vector)
# ─────────────────────────────────────────────────────────────────────────────

class TestStyleRewardVector:

    def test_returns_4d_array(self, default_hp):
        r = compute_style_rewards(
            j_lon=0.0, j_lat=0.0,
            v_ego=10.0, v_des=10.0, a_ego=1.0,
            lane_index=0, n_lanes=1,
            d_lat=0.0, delta_psi=0.0,
            d_lead=50.0, v_lead=0.0, has_lead=False,
            hp=default_hp,
        )
        assert r.shape == (4,)
        assert r.dtype == np.float32

    def test_all_components_in_zero_one(self, default_hp):
        r = compute_style_rewards(
            j_lon=1.0, j_lat=0.5,
            v_ego=8.0, v_des=10.0, a_ego=1.5,
            lane_index=1, n_lanes=3,
            d_lat=0.1, delta_psi=0.05,
            d_lead=20.0, v_lead=5.0, has_lead=True,
            hp=default_hp,
        )
        for i, val in enumerate(r):
            assert 0.0 < val <= 1.0, f"Component {i} = {val} out of (0,1]"


# ─────────────────────────────────────────────────────────────────────────────
# ZtNormaliser
# ─────────────────────────────────────────────────────────────────────────────

class TestZtNormaliser:

    def test_initial_normalisation_with_zero_input(self):
        norm = ZtNormaliser(reward_dim=4, beta=0.01)
        z = np.zeros(4)
        z_norm = norm.normalise(z)
        assert z_norm.shape == (4,)
        # With zero mean and unit var initialisation: normalised = 0
        assert np.allclose(z_norm, 0.0, atol=0.1)

    def test_update_shifts_mean(self):
        norm = ZtNormaliser(reward_dim=4, beta=0.1)
        for _ in range(100):
            norm.update(np.ones(4) * 5.0)
        # Mean should be near 5 now
        assert np.allclose(norm._mu, 5.0, atol=0.5)

    def test_normalise_output_clipped(self):
        norm = ZtNormaliser(reward_dim=4, beta=0.01)
        z = np.ones(4) * 1000.0  # extreme value
        z_norm = norm.normalise(z)
        assert np.all(z_norm <= 5.0)
        assert np.all(z_norm >= -5.0)

    def test_state_dict_roundtrip(self):
        norm = ZtNormaliser(reward_dim=4, beta=0.05)
        for _ in range(50):
            norm.update(np.random.randn(4))
        sd = norm.state_dict()
        norm2 = ZtNormaliser.from_state_dict(sd, reward_dim=4)
        assert np.allclose(norm._mu, norm2._mu)
        assert np.allclose(norm._var, norm2._var)


# ─────────────────────────────────────────────────────────────────────────────
# AlphaSchedule
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaSchedule:

    def test_starts_at_alpha_start(self, default_hp):
        sched = AlphaSchedule(default_hp, alpha_start=0.2, alpha_end=0.95, n_curriculum=1000)
        alpha, _ = sched.get(0)
        assert alpha == pytest.approx(0.2, abs=1e-6)

    def test_reaches_alpha_end(self, default_hp):
        sched = AlphaSchedule(default_hp, alpha_start=0.2, alpha_end=0.95, n_curriculum=1000)
        alpha, _ = sched.get(1000)
        assert alpha == pytest.approx(0.95, abs=1e-6)

    def test_monotone_increasing(self, default_hp):
        sched = AlphaSchedule(default_hp, alpha_start=0.2, alpha_end=0.95, n_curriculum=1000)
        alphas = [sched.get(n)[0] for n in range(0, 1001, 100)]
        for a, b in zip(alphas, alphas[1:]):
            assert b >= a

    def test_clamped_after_curriculum(self, default_hp):
        sched = AlphaSchedule(default_hp, alpha_start=0.2, alpha_end=0.95, n_curriculum=1000)
        alpha_end, _ = sched.get(1000)
        alpha_over, _ = sched.get(5000)
        assert alpha_end == pytest.approx(alpha_over, abs=1e-6)

    def test_compute_alpha_formula(self):
        alpha = compute_alpha(n=500, alpha_start=0.2, alpha_end=0.9, n_curriculum=1000)
        expected = 0.2 + (0.9 - 0.2) * 0.5
        assert alpha == pytest.approx(expected, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# CVaR estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestCVaR:

    def test_cvar_from_episodes_basic(self):
        from prism.morl.cvar_lagrangian import cvar_from_episodes
        costs = list(range(1, 101))  # 1..100
        # CVaR at alpha=0.95: top 5% = [96..100], mean = 98
        cvar = cvar_from_episodes(costs, alpha=0.95)
        assert cvar == pytest.approx(98.0, abs=1.0)

    def test_cvar_at_alpha_zero_is_mean(self):
        # CVaR_0 = expected value over the FULL distribution = mean.
        # (alpha=0 means "top 100% tail", not just the single worst.)
        from prism.morl.cvar_lagrangian import cvar_from_episodes
        costs = [1, 5, 3, 9, 2]
        cvar = cvar_from_episodes(costs, alpha=0.0)
        assert cvar == pytest.approx(float(np.mean(costs)), rel=1e-6)

    def test_cvar_at_high_alpha_is_single_worst(self):
        # alpha=0.8, N=5: tail fraction = 1-0.8 = 0.2 → ceil(0.2*5) = 1 episode.
        # CVaR = mean of top-1 = max.
        from prism.morl.cvar_lagrangian import cvar_from_episodes
        costs = [1, 5, 3, 9, 2]
        cvar = cvar_from_episodes(costs, alpha=0.8)
        assert cvar == pytest.approx(float(max(costs)), rel=1e-6)

    def test_lagrangian_update_increases_lambda_when_violated(self):
        from prism.morl.cvar_lagrangian import CVaRLagrangian
        lag = CVaRLagrangian(eta_lambda=0.1)
        # Add high costs so CVaR exceeds epsilon
        for _ in range(100):
            lag.add_episode(100.0)
        lam, cvar = lag.update_lambda(alpha=0.95, epsilon=1.0)
        assert lam > 0.0  # lambda should increase

    def test_lagrangian_lambda_clamped_non_negative(self):
        from prism.morl.cvar_lagrangian import CVaRLagrangian
        lag = CVaRLagrangian(eta_lambda=0.1, lambda_init=0.0)
        # Add zero costs so CVaR < epsilon
        for _ in range(100):
            lag.add_episode(0.0)
        lam, _ = lag.update_lambda(alpha=0.95, epsilon=50.0)
        assert lam >= 0.0  # never negative

    def test_compute_episode_cost(self):
        from prism.morl.cvar_lagrangian import compute_episode_cost
        costs = [1.0, 1.0, 1.0]
        gamma = 0.99
        expected = 1.0 + 0.99 + 0.99**2
        assert compute_episode_cost(costs, gamma) == pytest.approx(expected, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
