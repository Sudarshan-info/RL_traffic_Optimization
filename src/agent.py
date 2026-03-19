"""
FILE: src/agent.py
PURPOSE: Q-Learning agent with epsilon-greedy action selection
"""

import numpy as np
import json
from pathlib import Path
from src.config_loader import get_section


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy action selection.
    All hyperparameters can be set in config.json under "agent" section.
    """

    # Load agent config once
    AGENT_CONFIG = get_section("agent", {})

    def __init__(
        self,
        state_size: int,
        n_actions: int,
        alpha: float = None,
        gamma: float = None,
        epsilon: float = None,
        epsilon_min: float = None,
        epsilon_decay: float = None,
    ):
        """
        Initialize Q-Learning agent with values from config.
        """
        # Use provided values, otherwise use config, otherwise use defaults
        self.alpha = alpha if alpha is not None else self.AGENT_CONFIG.get("alpha", 0.3)
        self.gamma = (
            gamma if gamma is not None else self.AGENT_CONFIG.get("gamma", 0.95)
        )
        self.epsilon = (
            epsilon
            if epsilon is not None
            else self.AGENT_CONFIG.get("epsilon_start", 1.0)
        )
        self.epsilon_min = (
            epsilon_min
            if epsilon_min is not None
            else self.AGENT_CONFIG.get("epsilon_min", 0.01)
        )
        self.epsilon_decay = (
            epsilon_decay
            if epsilon_decay is not None
            else self.AGENT_CONFIG.get("epsilon_decay", 0.995)
        )

        self.q_table = np.zeros((state_size, n_actions))

    def choose_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        """Update Q-value using Q-learning rule."""
        best_next = 0.0 if done else float(np.max(self.q_table[s_next]))
        td_target = r + self.gamma * best_next
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "models/q_table.npy"):
        """Save Q-table and metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

        meta = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": float(self.epsilon),
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        Path(path.replace(".npy", "_meta.json")).write_text(json.dumps(meta, indent=2))
        print(f"Model saved -> {path}")

    def load(self, path: str = "models/q_table.npy"):
        """Load Q-table from file."""
        self.q_table = np.load(path)
        print(f"Model loaded <- {path}")

        # Try to load metadata
        try:
            meta_path = path.replace(".npy", "_meta.json")
            if Path(meta_path).exists():
                meta = json.loads(Path(meta_path).read_text())
                self.alpha = meta.get("alpha", self.alpha)
                self.gamma = meta.get("gamma", self.gamma)
                self.epsilon = meta.get("epsilon", self.epsilon)
                self.epsilon_min = meta.get("epsilon_min", self.epsilon_min)
                self.epsilon_decay = meta.get("epsilon_decay", self.epsilon_decay)
        except:
            pass
