import logging
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import (
        recall_score,
        precision_score,
        f1_score,
        roc_auc_score,
        log_loss,
        brier_score_loss,
        average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


def _compute_ice(targets, probs_2d, n_bins=10):
    """Compute Integrated Calibration Error (simplified)."""
    if len(targets) == 0:
        return 1.0
    pos_probs = probs_2d[:, 1]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ice = 0.0
    for i in range(n_bins):
        mask = (pos_probs >= bin_edges[i]) & (pos_probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (pos_probs >= bin_edges[i]) & (pos_probs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_pred = pos_probs[mask].mean()
        avg_true = targets[mask].mean()
        ice += np.abs(avg_pred - avg_true) * mask.sum()
    return ice / len(targets)


def report_model_perf(
    targets,
    columns=None,
    df=None,
    model_name="",
    model=None,
    target_label_encoder=None,
    preds=None,
    probs=None,
    plot_file=None,
    metrics=None,
    group_ids=None,
    **kwargs,
):
    """Minimal stub for mlframe report_model_perf.

    Computes standard classification metrics and stores them in metrics[1].
    """
    if metrics is None:
        metrics = {}

    targets = np.asarray(targets).ravel()
    preds = np.asarray(preds).ravel() if preds is not None else (probs[:, 1] >= 0.5).astype(np.int8)
    pos_probs = np.asarray(probs[:, 1]).ravel() if probs.ndim == 2 else np.asarray(probs).ravel()

    if SKLEARN_AVAILABLE and len(targets) > 0:
        try:
            recall = float(recall_score(targets, preds, zero_division=0))
        except Exception:
            recall = 0.0
        try:
            precision = float(precision_score(targets, preds, zero_division=0))
        except Exception:
            precision = 0.0
        try:
            f1 = float(f1_score(targets, preds, zero_division=0))
        except Exception:
            f1 = 0.0
        try:
            roc_auc = float(roc_auc_score(targets, pos_probs))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(targets, pos_probs))
        except Exception:
            pr_auc = 0.0
        try:
            ll = float(log_loss(targets, probs))
        except Exception:
            ll = 1.0
        try:
            brier = float(brier_score_loss(targets, pos_probs))
        except Exception:
            brier = 0.25
    else:
        recall = precision = f1 = 0.0
        roc_auc = 0.5
        pr_auc = 0.0
        ll = 1.0
        brier = 0.25

    try:
        ice = float(_compute_ice(targets, probs))
    except Exception:
        ice = 1.0

    metrics[1] = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": ll,
        "brier_loss": brier,
        "ice": ice,
        "calibration_mae": ice,
        "calibration_std": 0.0,
        "class_integral_error": 0.0,
    }

    return None, None
