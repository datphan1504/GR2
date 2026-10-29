"""Model Trainer managing full training loops, validation, and prediction exports."""

import math
import os
import time
from typing import Dict, Any, List

import numpy as np

from src.data.dataset import generate, get_split_size
from src.models.architectures import build_model
from src.training.scheduler import get_callbacks
from src.utils.helpers import get_logger, ensure_dir

logger = get_logger("Trainer")


class Trainer:
    """Manages training and prediction lifecycle for website fingerprinting models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_mon_sites = config['num_mon_sites']
        self.num_mon_inst_train = config['num_mon_inst_train']
        self.num_mon_inst_test = config['num_mon_inst_test']
        self.num_unmon_sites_train = config.get('num_unmon_sites_train', 0)
        self.num_unmon_sites_test = config.get('num_unmon_sites_test', 0)

        self.batch_size = config.get('batch_size', 50)
        self.model_name = config.get('model_name', 'var-cnn').lower()
        self.epochs = (
            config.get('var_cnn_max_epochs', 150)
            if self.model_name == 'var-cnn'
            else config.get('df_epochs', 30)
        )

        self.predictions_dir = ensure_dir(config.get('predictions_dir', 'artifacts/predictions/'))
        self.artifacts_dir = ensure_dir(config.get('artifacts_dir', 'artifacts/'))
        self.checkpoint_dir = ensure_dir(config.get('checkpoint_dir', os.path.join(self.artifacts_dir, 'checkpoints')))

    def train_and_validate(self, model, mixture_num: int, sub_model_name: str) -> None:
        """Trains and validates a given sub-model."""
        logger.info(f"--- Training {self.model_name} sub-model: {sub_model_name} ---")

        train_size = get_split_size(self.config, 'training_data')
        val_size = get_split_size(self.config, 'validation_data')

        train_steps = max(1, math.ceil(train_size / self.batch_size))
        val_steps = max(1, math.ceil(val_size / self.batch_size))

        callbacks = get_callbacks(self.config, sub_model_name)

        train_gen = generate(self.config, 'training_data', mixture_num)
        val_gen = generate(self.config, 'validation_data', mixture_num)

        start_time = time.time()
        model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=val_steps
        )

        duration = time.time() - start_time
        logger.info(f"Completed training {sub_model_name} in {duration:.2f} seconds.")

    def predict_and_save(self, model, mixture_num: int, sub_model_name: str) -> str:
        """Computes and saves predictions on the test set."""
        logger.info(f"Generating test predictions for {self.model_name} ({sub_model_name})...")

        best_weights_path = os.path.join(self.checkpoint_dir, f"{sub_model_name}_best.weights.h5")
        if os.path.exists(best_weights_path):
            logger.info(f"Loading best weights from {best_weights_path}")
            model.load_weights(best_weights_path)

        test_size = get_split_size(self.config, 'test_data')
        test_steps = max(1, math.ceil(test_size / self.batch_size))

        test_gen = generate(self.config, 'test_data', mixture_num)

        start_time = time.time()
        predictions = model.predict(test_gen, steps=test_steps, verbose=0)
        predictions = predictions[:test_size]

        output_pred_path = os.path.join(self.predictions_dir, f"{sub_model_name}_model.npy")
        np.save(output_pred_path, predictions)

        duration = time.time() - start_time
        logger.info(f"Saved predictions ({predictions.shape}) to {output_pred_path} in {duration:.2f}s.")
        return output_pred_path

    def run(self) -> List[str]:
        """Runs the entire training and prediction pipeline for all configured mixtures."""
        mixture = self.config.get('mixture', [["dir", "metadata"]])
        prediction_files = []

        for mixture_num, inner_comb in enumerate(mixture):
            sub_model_name = '_'.join(inner_comb)
            model = build_model(self.config, mixture_num)
            self.train_and_validate(model, mixture_num, sub_model_name)
            pred_path = self.predict_and_save(model, mixture_num, sub_model_name)
            prediction_files.append(pred_path)

        return prediction_files
