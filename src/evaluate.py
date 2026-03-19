"""
FILE: src/evaluate.py
PURPOSE: Compare RL agent vs fixed timer vs random agent
"""

import numpy as np
from src.environment import TrafficEnvironment
from src.agent import QLearningAgent
from src.config_loader import get_section


# Load evaluation config
EVAL_CONFIG = get_section("evaluation", {})


def run_policy(env, policy_fn, n_episodes=None, max_steps=None):
    """Run a policy for multiple episodes and collect stats."""
    if n_episodes is None:
        n_episodes = EVAL_CONFIG.get("n_episodes", 100)
    if max_steps is None:
        max_steps = EVAL_CONFIG.get("max_steps", 50)

    total_rewards, total_queues = [], []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_queue = 0
        steps = 0

        for _ in range(max_steps):
            action = policy_fn(state)
            state, reward, done = env.step(action)
            episode_reward += reward
            episode_queue += env.queue_north + env.queue_south
            steps += 1
            if done:
                break

        total_rewards.append(episode_reward)
        total_queues.append(episode_queue / steps)

    return {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_queue": float(np.mean(total_queues)),
        "std_queue": float(np.std(total_queues)),
    }


def evaluate(data_path="data/traffic_data.csv", n_episodes=None):
    """Compare RL agent vs fixed timer vs random agent."""

    if n_episodes is None:
        n_episodes = EVAL_CONFIG.get("n_episodes", 100)

    max_steps = EVAL_CONFIG.get("max_steps", 50)
    fixed_action = EVAL_CONFIG.get("fixed_timer_action", 2)

    print("\n" + "=" * 60)
    print("EVALUATION - Comparing Controllers")
    print("=" * 60)
    print(f"Episodes: {n_episodes} | Max steps: {max_steps}")

    env = TrafficEnvironment(data_path)

    # RL Agent
    agent = QLearningAgent(env.state_size, env.N_ACTIONS)
    try:
        agent.load()
        agent.epsilon = 0.0
        print("✓ RL Agent loaded")
    except:
        print("⚠ No trained model found, using untrained agent")

    rl_results = run_policy(
        env, lambda s: agent.choose_action(s), n_episodes, max_steps
    )
    fixed_results = run_policy(env, lambda s: fixed_action, n_episodes, max_steps)
    random_results = run_policy(
        env, lambda s: np.random.randint(env.N_ACTIONS), n_episodes, max_steps
    )

    results = {
        "RL Agent": rl_results,
        "Fixed Timer": fixed_results,
        "Random Agent": random_results,
    }

    # Calculate improvements
    fixed_queue = fixed_results["mean_queue"]
    random_queue = random_results["mean_queue"]
    rl_queue = rl_results["mean_queue"]

    rl_vs_fixed = (
        ((fixed_queue - rl_queue) / fixed_queue * 100) if fixed_queue > 0 else 0
    )
    rl_vs_random = (
        ((random_queue - rl_queue) / random_queue * 100) if random_queue > 0 else 0
    )

    print("\nResults:")
    print(
        f"{'Controller':<15} {'Reward':>10} {'Queue':>10} {'vs Fixed':>10} {'vs Random':>10}"
    )
    print(
        f"{'RL Agent':<15} {rl_results['mean_reward']:>10.2f} {rl_queue:>10.2f} {rl_vs_fixed:>9.1f}% {rl_vs_random:>9.1f}%"
    )
    print(
        f"{'Fixed Timer':<15} {fixed_results['mean_reward']:>10.2f} {fixed_queue:>10.2f} {'0.0%':>9} {((random_queue - fixed_queue)/random_queue*100):>9.1f}%"
    )
    print(
        f"{'Random Agent':<15} {random_results['mean_reward']:>10.2f} {random_queue:>10.2f} {((fixed_queue - random_queue)/fixed_queue*100):>9.1f}% {'0.0%':>9}"
    )

    if rl_vs_fixed > 20:
        print(f"\n✓ RL outperforms fixed timer by {rl_vs_fixed:.1f}%")
    elif rl_vs_fixed > 0:
        print(f"\n✓ RL slightly better than fixed timer (+{rl_vs_fixed:.1f}%)")
    else:
        print(f"\n⚠ RL needs more training ({rl_vs_fixed:.1f}% worse than fixed)")

    return results
