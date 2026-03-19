"""
FILE: src/config.py
PURPOSE: Load and validate configuration from JSON
UPDATED: Now includes ALL configuration sections
"""

import json
from pathlib import Path


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"✅ Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️ Config file not found. Using defaults.")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing config: {e}")
        return get_default_config()


def get_default_config():
    """Return default configuration with ALL sections."""
    return {
        "project": {"name": "Traffic Signal RL", "version": "3.0", "random_seed": 42},
        "data": {"n_days": 30, "n_intersections": 5, "base_arrival": 8},
        "agent": {
            "alpha": 0.3,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
        },
        "environment": {
            "n_queue_bins": 10,
            "n_hour_bins": 6,
            "max_steps_per_episode": 50,
            "max_queue": 50,
            "discharge_rate": 0.5,
            "queue_clamp_min": 5,
            "queue_clamp_max": 25,
            "reward_scale": 10.0,
            "green_times": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        },
        "training": {
            "n_episodes": 1000,
            "log_interval": 50,
            "test_interval": 100,
            "test_episodes": 10,
            "save_model": True,
        },
        "ml_models": {
            "train": True,
            "test_size": 0.2,
            "random_forest": {"n_estimators": 100},
            "svr": {"C": 10, "epsilon": 0.5},
        },
        "evaluation": {"n_episodes": 100, "max_steps": 50, "fixed_timer_action": 2},
        "visualization": {
            "smoothing_window": 50,
            "figure_dpi": 150,
            "style": "whitegrid",
        },
        "reporting": {
            "generate_pdf": True,
            "include_charts": True,
            "save_metrics_csv": True,
        },
    }


def get_config_section(config, section, default=None):
    """Safely get a section from config with default fallback."""
    if default is None:
        default = {}
    return config.get(section, default)


def get_config_value(config, section, key, default=None):
    """Safely get a value from config with default fallback."""
    try:
        return config[section][key]
    except (KeyError, TypeError):
        return default


def print_config_summary(config):
    """Print a nice summary of current configuration."""
    print("\n" + "=" * 60)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 60)

    # Project Info
    project = config.get("project", {})
    print(f"\n📁 PROJECT:")
    print(f"   Name: {project.get('name', 'Traffic RL')}")
    print(f"   Version: {project.get('version', '1.0')}")
    print(f"   Random Seed: {project.get('random_seed', 42)}")

    # Data Section
    data = config.get("data", {})
    print(f"\n📊 DATA:")
    n_days = data.get("n_days", 30)
    n_intersections = data.get("n_intersections", 5)
    print(f"   Days: {n_days} | Intersections: {n_intersections}")
    print(f"   Total rows: {n_days * 24 * 12 * n_intersections:,}")

    # Agent Section
    agent = config.get("agent", {})
    print(f"\n🤖 RL AGENT:")
    print(f"   Learning rate (alpha): {agent.get('alpha', 0.3)}")
    print(f"   Discount (gamma): {agent.get('gamma', 0.95)}")
    print(f"   Exploration decay: {agent.get('epsilon_decay', 0.995)}")
    print(f"   Min exploration: {agent.get('epsilon_min', 0.01)}")

    # Environment Section
    env = config.get("environment", {})
    print(f"\n🌍 ENVIRONMENT:")
    print(f"   Queue bins: {env.get('n_queue_bins', 10)}")
    print(f"   Hour bins: {env.get('n_hour_bins', 6)}")
    print(f"   Max steps per episode: {env.get('max_steps_per_episode', 50)}")
    print(
        f"   Queue clamp: {env.get('queue_clamp_min', 5)}-{env.get('queue_clamp_max', 25)}"
    )

    # Training Section
    training = config.get("training", {})
    print(f"\n🏋️ TRAINING:")
    n_episodes = training.get("n_episodes", 1000)
    max_steps = env.get("max_steps_per_episode", 50)
    print(f"   Episodes: {n_episodes}")
    print(f"   Log interval: {training.get('log_interval', 50)}")
    print(f"   Test interval: {training.get('test_interval', 100)}")
    print(f"   Total steps: {n_episodes * max_steps:,}")

    # ML Models Section
    ml = config.get("ml_models", {})
    print(f"\n🤖 ML MODELS:")
    print(f"   Train ML: {ml.get('train', True)}")
    print(f"   Test size: {ml.get('test_size', 0.2)}")
    rf = ml.get("random_forest", {})
    print(f"   Random Forest trees: {rf.get('n_estimators', 100)}")

    # Evaluation Section
    eval_config = config.get("evaluation", {})
    print(f"\n📈 EVALUATION:")
    print(f"   Test episodes: {eval_config.get('n_episodes', 100)}")
    print(f"   Max steps: {eval_config.get('max_steps', 50)}")
    print(f"   Fixed timer action: {eval_config.get('fixed_timer_action', 2)}")

    # Visualization Section
    viz = config.get("visualization", {})
    print(f"\n🎨 VISUALIZATION:")
    print(f"   Smoothing window: {viz.get('smoothing_window', 50)}")
    print(f"   Figure DPI: {viz.get('figure_dpi', 150)}")
    print(f"   Style: {viz.get('style', 'whitegrid')}")

    print("\n" + "=" * 60)


# Quick test
if __name__ == "__main__":
    config = load_config()
    print_config_summary(config)

    # Test helper functions
    print(f"\n✅ Helper functions test:")
    print(f"   Agent alpha: {get_config_value(config, 'agent', 'alpha', 0.3)}")
    print(f"   Missing value: {get_config_value(config, 'missing', 'key', 'default')}")
