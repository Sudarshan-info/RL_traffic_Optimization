"""
FILE: src/visualize.py
PURPOSE: Create clean, presentation-ready charts from training logs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Create results directory
Path("results").mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


def load_training_history(history_path="logs/training_history.json"):
    """Load training history with error handling."""
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        n_episodes = len(history.get("episode_rewards", []))
        print(f"   ✓ Loaded {n_episodes} episodes")
        return history
    except:
        print(f"   ⚠ No training history found")
        return None


def plot_learning_curve(history_path="logs/training_history.json"):
    """Plot training progress."""
    history = load_training_history(history_path)
    if not history:
        return

    rewards = history["episode_rewards"]
    eps = history["epsilon_values"]
    n_episodes = len(rewards)

    # Smooth rewards
    window = min(50, n_episodes // 10)
    smooth = pd.Series(rewards).rolling(window, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot 1: Learning curve
    ax1.plot(rewards, alpha=0.2, color="gray", linewidth=0.5)
    ax1.plot(smooth, color="blue", lw=2)
    ax1.set_ylabel("Reward")
    ax1.set_title(f"Learning Progress ({n_episodes} episodes)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Exploration decay
    ax2.plot(eps, color="orange", lw=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Exploration Rate")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved: results/learning_curve.png")


def plot_step_by_step(history_path="logs/training_history.json", steps=5):
    """Show learning snapshots at different stages."""
    history = load_training_history(history_path)
    if not history:
        return

    rewards = history["episode_rewards"]
    n_episodes = len(rewards)

    # Pick checkpoints
    checkpoints = np.linspace(0, n_episodes - 1, steps + 1, dtype=int)[1:]

    fig, axes = plt.subplots(1, steps, figsize=(15, 3))
    fig.suptitle("Learning Progress at Different Stages", fontsize=14)

    for i, (ax, cp) in enumerate(zip(axes, checkpoints)):
        start = max(0, cp - 100)
        recent = rewards[start:cp]

        ax.plot(recent, color="blue", lw=1.5)
        ax.axhline(y=np.mean(recent), color="red", ls="--", alpha=0.7)
        ax.set_title(f"Episode {cp}")
        ax.set_xlabel(f"Last {len(recent)} eps")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Reward")

    plt.tight_layout()
    plt.savefig("results/step_by_step_learning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved: results/step_by_step_learning.png")


def plot_evaluation_comparison(eval_results):
    """Compare controllers - SIMPLIFIED: only queue lengths."""
    names = list(eval_results.keys())
    queues = [eval_results[n]["mean_queue"] for n in names]
    queue_err = [eval_results[n].get("std_queue", 0) for n in names]

    # Colors
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar chart
    bars = ax.bar(
        names, queues, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )

    # Add error bars
    if any(queue_err):
        ax.errorbar(names, queues, yerr=queue_err, fmt="none", color="black", capsize=5)

    # Labels and title
    ax.set_ylabel("Average Queue Length (cars)")
    ax.set_title("Controller Performance Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, queues):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.2,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add improvement annotation
    if "RL Agent" in eval_results and "Fixed Timer" in eval_results:
        rl_queue = eval_results["RL Agent"]["mean_queue"]
        fixed_queue = eval_results["Fixed Timer"]["mean_queue"]
        improvement = (fixed_queue - rl_queue) / fixed_queue * 100

        ax.text(
            0.5,
            -0.15,
            f"RL improves by {improvement:.1f}% vs Fixed Timer",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig("results/evaluation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved: results/evaluation_comparison.png")


def plot_ml_comparison(ml_results):
    """Compare ML models - SIMPLIFIED."""
    names = list(ml_results.keys())
    maes = [ml_results[n]["mae"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Horizontal bar chart
    bars = ax.barh(
        names, maes, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5
    )

    ax.set_xlabel("Mean Absolute Error")
    ax.set_title(
        "ML Models Performance (lower is better)", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, maes):
        ax.text(
            val + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("results/ml_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved: results/ml_model_comparison.png")


if __name__ == "__main__":
    print("\n📊 Generating charts...")

    plot_learning_curve()
    plot_step_by_step()

    # Try to load and plot evaluation results
    try:
        eval_path = Path("logs/evaluation_results.json")
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_results = json.load(f)
            plot_evaluation_comparison(eval_results)
    except Exception as e:
        print(f"   ⚠ Evaluation chart failed: {e}")

    # Try to load and plot ML results
    try:
        ml_path = Path("logs/ml_metrics.json")
        if ml_path.exists():
            with open(ml_path, "r") as f:
                ml_results = json.load(f)
            plot_ml_comparison(ml_results)
    except Exception as e:
        print(f"   ⚠ ML chart failed: {e}")

    print("\n   ✅ All charts saved in results/ folder")
