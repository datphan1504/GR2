"""Evaluation runner aggregating predictions and saving benchmark results."""

import os
import json
from typing import Dict, Any

import h5py
import numpy as np

from src.evaluation.metrics import log_cw, log_ow
from src.utils.helpers import get_logger, ensure_dir

logger = get_logger("Evaluation")


def evaluate_predictions(config: Dict[str, Any], output_result_file: str = None) -> Dict[str, Any]:
    """
    Evaluates individual sub-models and computes their ensemble performance.

    Args:
        config: Project configuration dictionary.
        output_result_file: Optional path for saving the JSON summary.

    Returns:
        results: Dictionary containing all evaluated metrics.
    """
    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config.get('num_unmon_sites_test', 0)
    num_unmon_sites_train = config.get('num_unmon_sites_train', 0)
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    data_dir = config.get('data_dir', 'data/processed/')
    predictions_dir = config.get('predictions_dir', 'artifacts/predictions/')
    artifacts_dir = ensure_dir(config.get('artifacts_dir', 'artifacts/'))
    mixture = config.get('mixture', [["dir", "metadata"]])

    h5_filename = f"{num_mon_sites}_{num_mon_inst}_{num_unmon_sites_train}_{num_unmon_sites_test}.h5"
    h5_path = os.path.join(data_dir, h5_filename)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Test labels file not found: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        test_labels = f['test_data/labels'][:]

    predictions = {}
    ensemble_softmax = None

    for inner_comb in mixture:
        sub_model_name = '_'.join(inner_comb)
        pred_file = os.path.join(predictions_dir, f"{sub_model_name}_model.npy")
        if not os.path.exists(pred_file):
            logger.warning(f"Prediction file not found: {pred_file}. Skipping sub-model {sub_model_name}.")
            continue

        softmax = np.load(pred_file)
        if ensemble_softmax is None:
            ensemble_softmax = np.zeros_like(softmax)
        ensemble_softmax += softmax
        predictions[sub_model_name] = softmax

    if not predictions:
        logger.error("No sub-model predictions were found to evaluate.")
        return {}

    ensemble_softmax /= len(predictions)
    if len(predictions) > 1:
        predictions['ensemble'] = ensemble_softmax

    parameters = {
        'actual_labels': test_labels,
        'num_mon_sites': num_mon_sites,
        'num_mon_inst_test': num_mon_inst_test,
        'num_unmon_sites_test': num_unmon_sites_test,
        'num_unmon_sites': num_unmon_sites
    }

    results = {}
    is_closed_world = (num_unmon_sites == 0)

    logger.info(f"Running {'Closed-World' if is_closed_world else 'Open-World'} Evaluation...")
    for sub_name, softmax_arr in predictions.items():
        if is_closed_world:
            log_cw(results, sub_name, softmax_arr, **parameters)
        else:
            log_ow(results, sub_name, softmax_arr, **parameters)

    if output_result_file is None:
        output_result_file = os.path.join(artifacts_dir, 'evaluation_results.json')

    with open(output_result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, sort_keys=True, indent=4)

    logger.info(f"Evaluation results successfully saved to {output_result_file}")
    return results


if __name__ == '__main__':
    from src.utils.helpers import load_config
    cfg = load_config()
    evaluate_predictions(cfg)
