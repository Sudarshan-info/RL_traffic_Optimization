"""
FILE: src/environment.py
PURPOSE: Simulate a traffic intersection with configurable parameters
"""

import numpy as np
import pandas as pd
from src.config_loader import get_section, get_value


class TrafficEnvironment:
    """
    Simulates a traffic intersection.

    State:   (queue_north_bin, queue_south_bin, hour_bin) -> single int
    Size:    n_queue_bins^2 * n_hour_bins states
    Actions: n_actions green-time choices from config
    Reward:  -(queue_north + queue_south) / reward_scale
    """

    # Load environment config once
    ENV_CONFIG = get_section("environment", {})

    # Get green times from config or use defaults
    GREEN_TIMES = ENV_CONFIG.get(
        "green_times", [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    )
    N_ACTIONS = len(GREEN_TIMES)

    def __init__(self, data_path: str = "data/traffic_data.csv"):
        self.df = pd.read_csv(data_path)

        # Load configuration
        self.n_queue_bins = self.ENV_CONFIG.get("n_queue_bins", 10)
        self.n_hour_bins = self.ENV_CONFIG.get("n_hour_bins", 6)
        self.max_queue = self.ENV_CONFIG.get("max_queue", 30)
        self.queue_clamp_min = self.ENV_CONFIG.get("queue_clamp_min", 5)
        self.queue_clamp_max = self.ENV_CONFIG.get("queue_clamp_max", 25)
        self.reward_scale = self.ENV_CONFIG.get("reward_scale", 10.0)

        # State size calculation
        self.state_size = self.n_queue_bins**2 * self.n_hour_bins

        # Cache queue bins once for performance
        self._bins = np.linspace(0, self.max_queue, self.n_queue_bins + 1)

        # Initialize state variables
        self.queue_north = 10.0
        self.queue_south = 10.0
        self.hour = 8
        self.reset()

    def _hour_bin(self, hour: int) -> int:
        """Convert hour to discrete bin based on time-of-day patterns."""
        if hour < 6:
            return 0
        elif hour < 7:
            return 1
        elif hour < 10:
            return 2
        elif hour < 16:
            return 3
        elif hour < 19:
            return 4
        else:
            return 5

    def _queue_bin(self, q: float) -> int:
        """Convert queue length to discrete bin."""
        bin_idx = int(np.digitize(q, self._bins)) - 1
        return min(bin_idx, self.n_queue_bins - 1)

    def _encode(self, qn: float, qs: float, hour: int) -> int:
        """Encode state as single integer."""
        bn = self._queue_bin(qn)
        bs = self._queue_bin(qs)
        bh = self._hour_bin(hour)
        return bn * self.n_queue_bins * self.n_hour_bins + bs * self.n_hour_bins + bh

    def reset(self) -> int:
        """Start a new episode with clamped initial queue."""
        row = self.df.sample(1).iloc[0]
        self.hour = int(row["hour"])

        base_q = float(
            np.clip(row["queue_length"], self.queue_clamp_min, self.queue_clamp_max)
        )

        self.queue_north = base_q
        self.queue_south = base_q * float(np.random.uniform(0.7, 1.3))

        return self._encode(self.queue_north, self.queue_south, self.hour)

    def step(self, action_idx: int):
        """Apply action and update environment."""
        green = self.GREEN_TIMES[action_idx]
        discharge = green * 0.5

        self.queue_north = max(
            0.0, self.queue_north - discharge / 2 + np.random.exponential(2)
        )
        self.queue_south = max(
            0.0, self.queue_south - discharge / 2 + np.random.exponential(2)
        )

        self.hour = (self.hour + 1) % 24
        reward = -(self.queue_north + self.queue_south) / self.reward_scale
        next_state = self._encode(self.queue_north, self.queue_south, self.hour)
        done = (self.queue_north + self.queue_south) < 1.0

        return next_state, reward, done
