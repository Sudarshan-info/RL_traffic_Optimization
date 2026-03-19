import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from src.environment import TrafficEnvironment
from src.agent import QLearningAgent


def train(
    n_episodes=1000, max_steps=50, data_path="data/traffic_data.csv", log_interval=50
):

    env = TrafficEnvironment(data_path)
    agent = QLearningAgent(env.state_size, env.N_ACTIONS)
    history = {"episode_rewards": [], "epsilon_values": [], "mean_queue_per_ep": []}

    print(f"Training {n_episodes} episodes ...")
    for ep in tqdm(range(n_episodes)):
        state = env.reset()
        total_rew = total_q = steps = 0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_s, rew, done = env.step(action)
            agent.update(state, action, rew, next_s, done)
            state = next_s
            total_rew += rew
            total_q += env.queue_north + env.queue_south
            steps += 1
            if done:
                break

        agent.decay_epsilon()
        history["episode_rewards"].append(total_rew)
        history["epsilon_values"].append(agent.epsilon)
        history["mean_queue_per_ep"].append(total_q / steps)

        if (ep + 1) % log_interval == 0:
            recent = np.mean(history["episode_rewards"][-log_interval:])
            print(
                f"  Ep {ep+1:>5} | Avg Reward: {recent:>8.2f} | Eps:{agent.epsilon:.4f}"
            )

    agent.save()
    log_path = Path("logs/training_history.json")
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(history))
    print("Training complete. Logs saved.")

    # Run quick evaluation and save results
    try:
        print("\n📊 Running quick evaluation...")
        from src.evaluate import evaluate

        eval_results = evaluate(n_episodes=50)

        # Save evaluation results
        eval_path = Path("logs/evaluation_results.json")

        # Convert numpy values to Python floats for JSON
        clean_results = {}
        for agent_name, metrics in eval_results.items():
            clean_results[agent_name] = {
                "mean_reward": float(metrics["mean_reward"]),
                "std_reward": float(metrics["std_reward"]),
                "mean_queue": float(metrics["mean_queue"]),
                "std_queue": float(metrics["std_queue"]),
            }

        eval_path.write_text(json.dumps(clean_results, indent=2))
        print(f"✅ Evaluation results saved to {eval_path}")
    except Exception as e:
        print(f"⚠️ Could not run evaluation: {e}")

    return history
