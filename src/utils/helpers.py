"""Utility functions for logging, seeding, configuration loading, and I/O."""

import os
import json
import random
import logging
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np


def get_logger(name: str = "Var-CNN", level: int = logging.INFO) -> logging.Logger:
    """Creates and returns a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(directory_path: str) -> str:
    """Ensures a directory exists, creating it if necessary."""
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    return directory_path


def load_file_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a JSON or YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    if ext in [".yaml", ".yml"]:
        if yaml is None:
            raise ImportError("PyYAML is required to parse YAML config files. Install with 'pip install pyyaml'")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {ext}. Use .yaml, .yml or .json")


def load_config(
    train_config_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
    base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads and merges training and model configuration files.
    If no paths provided, falls back to default paths in configs/.
    """
    if base_dir is None:
        # Assuming GR2 root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    if train_config_path is None:
        train_config_path = os.path.join(base_dir, "configs", "train_config.yaml")
        if not os.path.exists(train_config_path):
            # Fallback to json if exists
            train_config_path = os.path.join(base_dir, "configs", "config.json")

    if model_config_path is None:
        model_config_path = os.path.join(base_dir, "configs", "model_config.yaml")

    config: Dict[str, Any] = {}

    if os.path.exists(train_config_path):
        config.update(load_file_config(train_config_path))

    if os.path.exists(model_config_path):
        config.update(load_file_config(model_config_path))

    # Normalize relative paths to base_dir
    for path_key in ["data_dir", "raw_data_dir", "artifacts_dir", "predictions_dir", "checkpoint_dir"]:
        if path_key in config and not os.path.isabs(config[path_key]):
            config[path_key] = os.path.normpath(os.path.join(base_dir, config[path_key]))
            ensure_dir(config[path_key])

    return config


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
            tf.random.set_seed(seed)
        elif hasattr(tf, "set_random_seed"):
            tf.set_random_seed(seed)
    except ImportError:
        pass
