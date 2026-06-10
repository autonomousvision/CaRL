"""
Plot the PRISM distributional Pareto front.

Loads the z_T vectors collected during evaluation (or training) and plots
pairwise 2D projections of the Pareto front across the 4 style objectives:
    0: Comfort
    1: Progress
    2: Lateral discipline
    3: Spacing

Usage
-----
    python scripts/visualise_pareto.py \\
        --pareto_points runs/prism_run_001/eval/pareto_points.npy \\
        --output_dir runs/prism_run_001/figures

    # Or using raw episode_zt data from eval JSON files:
    python scripts/visualise_pareto.py \\
        --eval_dir runs/prism_run_001/eval \\
        --output_dir runs/prism_run_001/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PRISM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PRISM_ROOT))

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

OBJECTIVE_NAMES = ["Comfort", "Progress", "Lateral", "Spacing"]
POLICY_COLORS = ["#E63946", "#2A9D8F", "#F4A261", "#457B9D", "#6A0572"]
POLICY_LABELS = [
    "Comfort-seeking",
    "Progress-oriented",
    "Disciplined",
    "Cautious",
    "Balanced",
]


def load_pareto_points(eval_dir: Path, n_policies: int = 5) -> np.ndarray:
    """Load per-policy mean z_T from eval JSON files.  Shape: (K, 4)."""
    points = []
    for k in range(n_policies):
        json_path = eval_dir / f"eval_results_{k}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            results = json.load(f)
        zts = [r["episode_zt"] for r in results if r.get("episode_zt") is not None]
        if zts:
            points.append(np.mean(zts, axis=0))
    return np.array(points) if points else np.empty((0, 4))


def plot_pairwise_pareto(
    pareto_points: np.ndarray,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Plot all 6 pairwise 2D projections of the 4D Pareto front."""
    output_dir.mkdir(parents=True, exist_ok=True)
    K = pareto_points.shape[0]

    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    n_pairs = len(pairs)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for ax_idx, (i, j) in enumerate(pairs):
        ax = axes_flat[ax_idx]
        for k in range(K):
            color = POLICY_COLORS[k % len(POLICY_COLORS)]
            label = POLICY_LABELS[k] if k < len(POLICY_LABELS) else f"Policy {k}"
            ax.scatter(
                pareto_points[k, i],
                pareto_points[k, j],
                color=color,
                label=label,
                s=120,
                zorder=5,
            )
            ax.annotate(
                label.split("-")[0],
                (pareto_points[k, i], pareto_points[k, j]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

        ax.set_xlabel(OBJECTIVE_NAMES[i], fontsize=12)
        ax.set_ylabel(OBJECTIVE_NAMES[j], fontsize=12)
        ax.set_title(f"{OBJECTIVE_NAMES[i]} vs {OBJECTIVE_NAMES[j]}", fontsize=13)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

    handles = [
        plt.scatter([], [], color=POLICY_COLORS[k % len(POLICY_COLORS)],
                    label=(POLICY_LABELS[k] if k < len(POLICY_LABELS) else f"Policy {k}"),
                    s=80)
        for k in range(K)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=K, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("PRISM Distributional Pareto Front (mean z_T per policy)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    save_path = output_dir / "pareto_front_pairwise.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved pairwise Pareto plot → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_radar(
    pareto_points: np.ndarray,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Radar / spider plot of mean z_T per policy across 4 objectives."""
    K = pareto_points.shape[0]
    N = 4  # objectives

    # Normalise to [0, 1] across policies
    max_vals = pareto_points.max(axis=0, keepdims=True) + 1e-8
    normalised = pareto_points / max_vals

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for k in range(K):
        values = normalised[k].tolist() + normalised[k][:1].tolist()
        color = POLICY_COLORS[k % len(POLICY_COLORS)]
        label = POLICY_LABELS[k] if k < len(POLICY_LABELS) else f"Policy {k}"
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(OBJECTIVE_NAMES, fontsize=13)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Style profile per policy (normalised)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    save_path = output_dir / "pareto_radar.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved radar plot → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualise PRISM Pareto front")
    parser.add_argument("--pareto_points", type=str, default=None,
                        help="Path to .npy file of shape (K, 4)")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Directory containing eval_results_k.json files")
    parser.add_argument("--n_policies", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.pareto_points:
        pareto_points = np.load(args.pareto_points)
    elif args.eval_dir:
        pareto_points = load_pareto_points(
            Path(args.eval_dir), n_policies=args.n_policies
        )
    else:
        print("Error: supply --pareto_points or --eval_dir")
        sys.exit(1)

    if pareto_points.shape[0] == 0:
        print("No Pareto points found. Run evaluation first.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    print(f"Pareto points shape: {pareto_points.shape}")
    print(f"Values:\n{pareto_points.round(4)}")

    plot_pairwise_pareto(pareto_points, output_dir, show=args.show)
    plot_radar(pareto_points, output_dir, show=args.show)


if __name__ == "__main__":
    main()
