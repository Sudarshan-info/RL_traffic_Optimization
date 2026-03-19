import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

Path("results").mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


def plot_learning_curve(history_path="logs/training_history.json"):
    h = json.loads(Path(history_path).read_text())
    rewards = h["episode_rewards"]
    eps = h["epsilon_values"]
    window = 50
    smooth = pd.Series(rewards).rolling(window).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(rewards, alpha=0.3, color="steelblue", label="Raw reward")
    ax1.plot(smooth, color="navy", lw=2, label=f"{window}-ep average")
    ax1.set_ylabel("Episode Reward")
    ax1.legend()
    ax1.set_title("Q-Learning: Reward Over Training")

    ax2.plot(eps, color="orange")
    ax2.set_ylabel("Epsilon")
    ax2.set_xlabel("Episode")
    ax2.set_title("Epsilon Decay (explores less over time)")

    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=150)
    print("Saved: results/learning_curve.png")
    plt.close()


def plot_step_by_step(history_path="logs/training_history.json", steps=5):
    h = json.loads(Path(history_path).read_text())
    rewards = h["episode_rewards"]
    n = len(rewards)
    checkpoints = np.linspace(0, n - 1, steps + 1, dtype=int)[1:]

    fig, axes = plt.subplots(1, steps, figsize=(18, 4))
    fig.suptitle("Step-by-Step: How the Agent Learns Over Time", fontsize=14)
    for i, (ax, cp) in enumerate(zip(axes, checkpoints)):
        sl = rewards[max(0, cp - 100) : cp]
        ax.plot(sl, color="steelblue", lw=1.2)
        ax.set_title(f"Episode ~{cp}\nAvg: {np.mean(sl):.1f}")
        ax.axhline(np.mean(sl), color="red", ls="--", alpha=0.6)
        ax.set_xlabel("Recent Episodes")
        if i == 0:
            ax.set_ylabel("Reward")

    plt.tight_layout()
    plt.savefig("results/step_by_step_learning.png", dpi=150)
    print("Saved: results/step_by_step_learning.png")
    plt.close()


def plot_evaluation_comparison(eval_results):
    names = list(eval_results.keys())
    rewards = [eval_results[n]["mean_reward"] for n in names]
    queues = [eval_results[n]["mean_queue"] for n in names]
    colors = ["#2563EB", "#DC2626", "#16A34A"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(names, rewards, color=colors)
    ax1.set_title("Mean Episode Reward (higher = better)")
    ax2.bar(names, queues, color=colors)
    ax2.set_title("Mean Queue Length (lower = better)")
    plt.tight_layout()
    plt.savefig("results/evaluation_comparison.png", dpi=150)
    print("Saved: results/evaluation_comparison.png")
    plt.close()


def plot_ml_comparison(ml_results):
    names = list(ml_results.keys())
    maes = [ml_results[n]["mae"] for n in names]
    r2s = [ml_results[n]["r2"] for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.barh(names, maes, color="steelblue")
    ax1.set_title("MAE - lower is better")
    ax1.set_xlabel("MAE")
    ax2.barh(names, r2s, color="teal")
    ax2.set_title("R2 Score - higher is better")
    ax2.set_xlabel("R2")
    plt.tight_layout()
    plt.savefig("results/ml_model_comparison.png", dpi=150)
    print("Saved: results/ml_model_comparison.png")
    plt.close()
