"""CLI script to run data preprocessing, model training, and evaluation."""

import os
import sys
import argparse

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import load_config, set_seed, get_logger
from src.data.preprocessing import preprocess_pipeline
from src.training.trainer import Trainer
from src.evaluation.evaluate import evaluate_predictions

logger = get_logger("RunTraining")


def parse_args():
    parser = argparse.ArgumentParser(description="Var-CNN / DF Training and Evaluation Pipeline")
    parser.add_argument("--train-config", type=str, default=None, help="Path to train_config.yaml")
    parser.add_argument("--model-config", type=str, default=None, help="Path to model_config.yaml")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip data preprocessing step")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing predictions")
    parser.add_argument("--model-name", type=str, default=None, choices=["var-cnn", "df"], help="Override model name")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(train_config_path=args.train_config, model_config_path=args.model_config)

    if args.model_name:
        config['model_name'] = args.model_name
        if args.model_name == 'df':
            config['mixture'] = [['dir']]
            config['batch_size'] = 128

    set_seed(config.get("random_seed", 42))

    if args.eval_only:
        logger.info("Executing evaluation only...")
        evaluate_predictions(config)
        return

    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config.get('num_unmon_sites_test', 0)
    num_unmon_sites_train = config.get('num_unmon_sites_train', 0)

    data_dir = config.get('data_dir', 'data/processed/')
    target_h5 = os.path.join(data_dir, f"{num_mon_sites}_{num_mon_inst}_{num_unmon_sites_train}_{num_unmon_sites_test}.h5")

    if not args.skip_preprocess and not os.path.exists(target_h5):
        logger.info(f"Target HDF5 file {target_h5} not found. Running data preprocessing pipeline...")
        preprocess_pipeline(config)
    elif os.path.exists(target_h5):
        logger.info(f"Using preprocessed dataset at: {target_h5}")

    logger.info("Starting model training...")
    trainer = Trainer(config)
    trainer.run()

    logger.info("Starting ensemble model evaluation...")
    evaluate_predictions(config)
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
