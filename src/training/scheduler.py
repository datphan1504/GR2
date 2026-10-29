"""Callbacks and learning rate scheduling logic."""

import os
from typing import List, Dict, Any

import numpy as np

try:
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
except ImportError:
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback

from src.utils.helpers import ensure_dir


def get_callbacks(config: Dict[str, Any], sub_model_name: str = "var_cnn") -> List[Callback]:
    """
    Constructs training callbacks:
    - ReduceLROnPlateau
    - EarlyStopping
    - ModelCheckpoint (using .weights.h5 suffix for Keras 3 compatibility)
    """
    base_patience = config.get('var_cnn_base_patience', 5)
    artifacts_dir = ensure_dir(config.get('artifacts_dir', 'artifacts/'))
    checkpoint_dir = ensure_dir(config.get('checkpoint_dir', os.path.join(artifacts_dir, 'checkpoints')))

    checkpoint_file = os.path.join(checkpoint_dir, f"{sub_model_name}_best.weights.h5")

    monitor_metric = 'val_accuracy'
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        monitor_metric = 'val_acc'

    lr_reducer = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=float(np.sqrt(0.1)),
        cooldown=0,
        patience=base_patience,
        min_lr=1e-5,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=2 * base_patience,
        verbose=1,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        checkpoint_file,
        monitor=monitor_metric,
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    return [lr_reducer, early_stopping, model_checkpoint]
