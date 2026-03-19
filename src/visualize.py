"""
FILE: src/visualize.py
PURPOSE: Create professional presentation-ready charts from training logs
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
        print(f"   ✓ Loaded {n_episodes} episodes from {history_path}")
        return history
    except FileNotFoundError:
        print(f"   ⚠ Training history not found at {history_path}")
        return None
    except json.JSONDecodeError:
        print(f"   ⚠ Invalid JSON in {history_path}")
        return None


def plot_learning_curve(history_path="logs/training_history.json"):
    """Plot training progress with episode count displayed."""
    history = load_training_history(history_path)
    if not history:
        print("   ⚠ Cannot plot learning curve - no data")
        return

    rewards = history["episode_rewards"]
    eps = history["epsilon_values"]
    n_episodes = len(rewards)

    # Adaptive smoothing window
    window = min(50, max(10, n_episodes // 20))
    smooth = pd.Series(rewards).rolling(window, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Rewards
    ax1.plot(rewards, alpha=0.3, color="steelblue", label="Raw reward", linewidth=0.5)
    ax1.plot(smooth, color="navy", lw=2, label=f"{window}-ep moving average")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Episode Reward")
    ax1.set_title(f"Q-Learning: Training Progress ({n_episodes} episodes)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Add annotation with final performance
    final_reward = np.mean(rewards[-50:]) if n_episodes > 50 else np.mean(rewards)
    final_queue = (
        np.mean(history["mean_queue_per_ep"][-50:])
        if n_episodes > 50
        else np.mean(history["mean_queue_per_ep"])
    )

    stats_text = f"Final (last 50 eps):\nReward: {final_reward:.3f}\nQueue: {final_queue:.2f} cars"
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Plot 2: Epsilon decay
    ax2.plot(eps, color="orange", lw=2)
    ax2.set_ylabel("Epsilon (exploration rate)")
    ax2.set_xlabel(f"Episode")
    ax2.set_title(f"Exploration Decay (final: {eps[-1]:.4f})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: results/learning_curve.png ({n_episodes} episodes)")


def plot_step_by_step(history_path="logs/training_history.json", steps=5):
    """Show learning snapshots at different stages."""
    history = load_training_history(history_path)
    if not history:
        print("   ⚠ Cannot plot step-by-step - no data")
        return

    rewards = history["episode_rewards"]
    n_episodes = len(rewards)

    # Pick evenly spaced checkpoints
    checkpoints = np.linspace(0, n_episodes - 1, steps + 1, dtype=int)[1:]

    fig, axes = plt.subplots(1, steps, figsize=(18, 4))
    fig.suptitle(
        f"Step-by-Step: Agent Learning Over Time ({n_episodes} episodes)",
        fontsize=14,
        fontweight="bold",
    )

    for i, (ax, cp) in enumerate(zip(axes, checkpoints)):
        # Show recent episodes up to this point
        start = max(0, cp - 100)
        recent = rewards[start:cp]

        if len(recent) > 0:
            ax.plot(recent, color="steelblue", lw=1.5)
            avg = np.mean(recent)
            ax.axhline(
                y=avg, color="red", lw=2, linestyle="--", label=f"Avg: {avg:.2f}"
            )
            ax.set_title(f"Episode ~{cp}")
            ax.set_xlabel(f"Last {len(recent)} eps")
            ax.legend(fontsize=8, loc="upper right")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Episode ~{cp}")

        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Reward")

    plt.tight_layout()
    plt.savefig("results/step_by_step_learning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: results/step_by_step_learning.png")


def plot_evaluation_comparison(eval_results):
    """Compare RL vs Fixed vs Random with proper formatting."""
    names = list(eval_results.keys())
    rewards = [eval_results[n]["mean_reward"] for n in names]
    queues = [eval_results[n]["mean_queue"] for n in names]

    # Get standard deviations if available
    reward_err = [eval_results[n].get("std_reward", 0) for n in names]
    queue_err = [eval_results[n].get("std_queue", 0) for n in names]

    # Professional color palette
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Rewards (higher is better)
    bars1 = ax1.bar(
        names, rewards, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    if any(reward_err):
        ax1.errorbar(
            names,
            rewards,
            yerr=reward_err,
            fmt="none",
            color="black",
            capsize=5,
            capthick=1,
        )

    ax1.set_title(
        "Mean Episode Reward (higher = better)", fontsize=13, fontweight="bold"
    )
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels with proper positioning
    for bar, val in zip(bars1, rewards):
        height = bar.get_height()
        if height < 0:
            # Negative values: label inside bar at top
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                color="white" if abs(height) > 1 else "black",
            )
        else:
            # Positive values: label above bar
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

    # Plot 2: Queue length (lower is better)
    bars2 = ax2.bar(
        names, queues, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    if any(queue_err):
        ax2.errorbar(
            names,
            queues,
            yerr=queue_err,
            fmt="none",
            color="black",
            capsize=5,
            capthick=1,
        )

    ax2.set_title("Mean Queue Length (lower = better)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Queue (cars)")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars2, queues):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.2,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Add improvement annotation
    if "RL Agent" in eval_results and "Fixed Timer" in eval_results:
        rl_queue = eval_results["RL Agent"]["mean_queue"]
        fixed_queue = eval_results["Fixed Timer"]["mean_queue"]
        improvement = (fixed_queue - rl_queue) / fixed_queue * 100

        # Determine performance verdict
        if improvement > 20:
            verdict = "🏆 RL SUPERIOR"
            color = "lightgreen"
        elif improvement > 0:
            verdict = "✓ RL BETTER"
            color = "wheat"
        else:
            verdict = "⚠ RL NEEDS TRAINING"
            color = "lightsalmon"

        ax2.text(
            0.5,
            -0.2,
            f"{verdict}: {improvement:.1f}% vs Fixed Timer",
            transform=ax2.transAxes,
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig("results/evaluation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: results/evaluation_comparison.png")

    # Print summary to console
    print("\n   📊 Evaluation Summary:")
    for name, q in zip(names, queues):
        print(f"      {name:12}: {q:.2f} cars")
    if "RL Agent" in eval_results and "Fixed Timer" in eval_results:
        print(f"      → Improvement: {improvement:.1f}%")


def plot_ml_comparison(ml_results):
    """Compare ML models with proper formatting."""
    names = list(ml_results.keys())
    maes = [ml_results[n]["mae"] for n in names]
    r2s = [ml_results[n]["r2"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MAE plot (lower is better)
    bars1 = ax1.barh(
        names, maes, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5
    )
    ax1.set_title(
        "Mean Absolute Error (lower = better)", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("MAE")

    # Add value labels
    for bar, val in zip(bars1, maes):
        ax1.text(
            val + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontweight="bold",
        )

    # R² plot (higher is better)
    bars2 = ax2.barh(
        names, r2s, color="teal", alpha=0.8, edgecolor="black", linewidth=0.5
    )
    ax2.set_title("R² Score (higher = better)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("R²")
    ax2.set_xlim(0, 1)  # R² is between 0 and 1

    # Add value labels
    for bar, val in zip(bars2, r2s):
        ax2.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("results/ml_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: results/ml_model_comparison.png")


if __name__ == "__main__":
    print("\n📊 Generating all charts...")

    # Try to load and plot each chart independently
    try:
        plot_learning_curve()
    except Exception as e:
        print(f"   ⚠ Learning curve failed: {e}")

    try:
        plot_step_by_step()
    except Exception as e:
        print(f"   ⚠ Step-by-step failed: {e}")

    try:
        # Try to load evaluation results
        eval_path = Path("logs/evaluation_results.json")
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_results = json.load(f)
            plot_evaluation_comparison(eval_results)
    except Exception as e:
        print(f"   ⚠ Evaluation comparison failed: {e}")

    try:
        # Try to load ML results
        ml_path = Path("logs/ml_metrics.json")
        if ml_path.exists():
            with open(ml_path, "r") as f:
                ml_results = json.load(f)
            plot_ml_comparison(ml_results)
    except Exception as e:
        print(f"   ⚠ ML comparison failed: {e}")

    print("\n   ✅ All charts generated in results/ folder")
