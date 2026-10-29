"""Evaluation metrics for closed-world and open-world website fingerprinting."""

from typing import Tuple, Dict, Any
import numpy as np

from src.utils.helpers import get_logger

logger = get_logger("Metrics")


def find_accuracy(
    model_predictions: np.ndarray,
    conf_thresh: float,
    actual_labels: np.ndarray,
    num_mon_sites: int,
    num_mon_inst_test: int,
    num_unmon_sites_test: int,
    num_unmon_sites: int
) -> Tuple[str, str, str]:
    """
    Computes two-class TPR, multi-class TPR, and FPR given a confidence threshold.
    """
    n_samples = min(len(model_predictions), len(actual_labels))
    model_predictions = model_predictions[:n_samples]
    actual_labels = actual_labels[:n_samples]

    true_classes = np.argmax(actual_labels, axis=1)
    thresh_model_labels = np.zeros(n_samples, dtype=int)

    for inst_num, softmax in enumerate(model_predictions):
        predicted_class = int(np.argmax(softmax))
        if predicted_class < num_mon_sites and softmax[predicted_class] < conf_thresh:
            thresh_model_labels[inst_num] = num_mon_sites
        else:
            thresh_model_labels[inst_num] = predicted_class

    two_class_true_pos = 0
    multi_class_true_pos = 0
    false_pos = 0

    for inst_num, inst_label in enumerate(true_classes):
        if inst_label == num_mon_sites:  # Unmonitored site
            if thresh_model_labels[inst_num] < num_mon_sites:
                false_pos += 1
        else:  # Monitored site
            if thresh_model_labels[inst_num] < num_mon_sites:
                two_class_true_pos += 1
            if thresh_model_labels[inst_num] == inst_label:
                multi_class_true_pos += 1

    actual_mon_test = int(np.sum(true_classes < num_mon_sites))
    total_mon_test = actual_mon_test if actual_mon_test > 0 else (num_mon_sites * num_mon_inst_test)

    two_class_tpr_val = (two_class_true_pos / total_mon_test * 100) if total_mon_test > 0 else 0.0
    multi_class_tpr_val = (multi_class_true_pos / total_mon_test * 100) if total_mon_test > 0 else 0.0

    two_class_tpr = f"{two_class_tpr_val:.2f}%"
    multi_class_tpr = f"{multi_class_tpr_val:.2f}%"

    actual_unmon_test = int(np.sum(true_classes == num_mon_sites))
    total_unmon_test = actual_unmon_test if actual_unmon_test > 0 else num_unmon_sites_test

    if num_unmon_sites == 0 or total_unmon_test == 0:
        fpr = "0.00%"
    else:
        fpr_val = false_pos / total_unmon_test * 100
        fpr = f"{fpr_val:.2f}%"

    return two_class_tpr, multi_class_tpr, fpr


def log_cw(results: Dict[str, Any], sub_model_name: str, softmax: np.ndarray, **parameters) -> None:
    """Logs closed-world evaluation metrics."""
    logger.info(f"{sub_model_name} model (Closed World):")
    _, multi_class_tpr, _ = find_accuracy(softmax, 0.0, **parameters)
    logger.info(f"\tAccuracy: {multi_class_tpr}")
    results[f"{sub_model_name}_acc"] = multi_class_tpr


def log_ow(results: Dict[str, Any], sub_model_name: str, softmax: np.ndarray, **parameters) -> None:
    """Logs open-world evaluation metrics across a sweep of confidence thresholds."""
    logger.info(f"{sub_model_name} model (Open World):")
    for conf_thresh in np.arange(0, 1.01, 0.1):
        two_class_tpr, multi_class_tpr, fpr = find_accuracy(softmax, conf_thresh, **parameters)
        logger.info(
            f"\tConf: {conf_thresh:.1f} | 2-class TPR: {two_class_tpr} | "
            f"Multi-class TPR: {multi_class_tpr} | FPR: {fpr}"
        )
        prefix = f"{sub_model_name}_{conf_thresh:.2f}"
        results[f"{prefix}_two_TPR"] = two_class_tpr
        results[f"{prefix}_multi_TPR"] = multi_class_tpr
        results[f"{prefix}_FPR"] = fpr
