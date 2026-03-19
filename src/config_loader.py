"""
FILE: src/config_loader.py
PURPOSE: Single source for config loading to avoid repeated file reads
"""

import json
from pathlib import Path
from src.config import load_config as load_config_full

# Global config cache
_CONFIG_CACHE = None


def get_config():
    """Get config singleton (loads once, caches for future calls)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config_full()
    return _CONFIG_CACHE


def get_section(section, default=None):
    """Get a specific section from config."""
    config = get_config()
    return config.get(section, default or {})


def get_value(section, key, default=None):
    """Get a specific value from config."""
    config = get_config()
    try:
        return config[section][key]
    except (KeyError, TypeError):
        return default
