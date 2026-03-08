"""
SpineIDS Clinical ML - Evaluation Metrics
====================================
Unified metric interface shared with the CNN pipeline to ensure
comparability when computing multimodal fusion results.

Contents:
- compute_all_metrics   : compute the full metric suite
- compute_calibration   : calibration metrics and curve data
- compute_dca           : decision curve analysis
- bootstrap_metrics     : bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix, precision_score, recall_score,
    brier_score_loss, roc_curve, precision_recall_curve, average_precision_score
)
import warnings

from .config import CLASS_NAMES, NUM_CLASSES, IDX_TO_CLASS


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute the full evaluation metric suite for a multi-class classifier.
    
    The metric definitions are identical to those in the CNN pipeline,
    ensuring comparability during multimodal fusion.
    
    Args:
        y_true: Ground-truth integer labels of shape (N,), values in {0, 1, 2}.
        y_prob: Predicted class probabilities of shape (N, 3); rows sum to 1.
        y_pred: Predicted class indices of shape (N,). Defaults to argmax(y_prob).
    
    Returns:
        Dictionary of metric name -> scalar value.  Keys include:
                Global : AUC_Macro, Accuracy, F1_Macro, Kappa, MCC, Brier_Score
                Per-class: {cls}_AUC, {cls}_Sens, {cls}_Spec, {cls}_Prec, {cls}_NPV, {cls}_F1
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    
    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = np.array(y_pred).astype(int)
    
    metrics = {}
    

    
    # AUC Macro (One-vs-Rest)
    try:
        metrics['AUC_Macro'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro'
        )
    except ValueError:
        metrics['AUC_Macro'] = np.nan
    
    # Accuracy
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 Macro
    metrics['F1_Macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Cohen's Kappa
    metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Matthews Correlation Coefficient
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    


    brier_scores = []
    for i in range(NUM_CLASSES):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        brier_scores.append(brier_score_loss(y_true_binary, y_prob_class))
    metrics['Brier_Score'] = np.mean(brier_scores)
    

    

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    for i, cls_name in enumerate(CLASS_NAMES):

        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        y_pred_binary = (y_pred == i).astype(int)
        
        # AUC (One-vs-Rest)
        try:
            metrics[f'{cls_name}_AUC'] = roc_auc_score(y_true_binary, y_prob_class)
        except ValueError:
            metrics[f'{cls_name}_AUC'] = np.nan
        

        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        # Sensitivity (Recall, TPR)
        metrics[f'{cls_name}_Sens'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (TNR)
        metrics[f'{cls_name}_Spec'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Precision (PPV)
        metrics[f'{cls_name}_Prec'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # NPV
        metrics[f'{cls_name}_NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # F1
        prec = metrics[f'{cls_name}_Prec']
        sens = metrics[f'{cls_name}_Sens']
        metrics[f'{cls_name}_F1'] = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute a (3 × 3) confusion matrix with optional normalisation.
    
    Args:
        y_true      : Ground-truth labels.
        y_pred   : Predicted labels.
        normalize: Normalisation mode – 'true', 'pred', 'all', or None.
    
    Returns:
        Confusion matrix of shape (3, 3).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    if normalize == 'true':
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype(float) / cm.sum()
    
    return cm


def compute_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Dict]:
    """
    Compute per-class ROC curve data (one-vs-rest).
    
    Args:
        y_true: Ground-truth labels of shape (N,).
        y_prob: Predicted class probabilities of shape (N, 3).
    
    Returns:
        Dict mapping class name to {fpr, tpr, thresholds, auc}.
    """
    roc_data = {}
    
    for i, cls_name in enumerate(CLASS_NAMES):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_class)
        auc = roc_auc_score(y_true_binary, y_prob_class)
        
        roc_data[cls_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        }
    
    return roc_data


def compute_pr_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Dict]:
    """
    Compute per-class precision-recall curve data (one-vs-rest).
    
    Args:
        y_true: Ground-truth labels of shape (N,).
        y_prob: Predicted class probabilities of shape (N, 3).
    
    Returns:
        Dict mapping class name to {precision, recall, thresholds, ap}.
    """
    pr_data = {}
    
    for i, cls_name in enumerate(CLASS_NAMES):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob_class)
        ap = average_precision_score(y_true_binary, y_prob_class)
        
        pr_data[cls_name] = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'ap': ap
        }
    
    return pr_data


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute calibration metrics (ECE, MCE) and binned curve data.
    
    Args:
        y_true: Ground-truth labels of shape (N,).
        y_prob: Predicted class probabilities of shape (N, 3).
        n_bins: Number of probability bins.
    
    Returns:
        Calibration data dictionary with keys:
                ECE, MCE (scalar), per_class (dict of binned arrays).
            
            
    """
    calibration_data = {
        'per_class': {}
    }
    
    ece_list = []
    mce_list = []
    
    for i, cls_name in enumerate(CLASS_NAMES):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob_class, bin_edges[1:-1])
        
        bin_means = []
        bin_accs = []
        bin_counts = []
        
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_means.append(y_prob_class[mask].mean())
                bin_accs.append(y_true_binary[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_means.append(np.nan)
                bin_accs.append(np.nan)
                bin_counts.append(0)
        
        bin_means = np.array(bin_means)
        bin_accs = np.array(bin_accs)
        bin_counts = np.array(bin_counts)
        

        valid_mask = ~np.isnan(bin_means) & ~np.isnan(bin_accs)
        if valid_mask.sum() > 0:
            ece = np.sum(bin_counts[valid_mask] * np.abs(bin_accs[valid_mask] - bin_means[valid_mask])) / np.sum(bin_counts[valid_mask])
            mce = np.max(np.abs(bin_accs[valid_mask] - bin_means[valid_mask]))
        else:
            ece = np.nan
            mce = np.nan
        
        ece_list.append(ece)
        mce_list.append(mce)
        
        calibration_data['per_class'][cls_name] = {
            'bin_means': bin_means,
            'bin_accs': bin_accs,
            'bin_counts': bin_counts,
            'ece': ece,
            'mce': mce
        }
    
    calibration_data['ECE'] = np.nanmean(ece_list)
    calibration_data['MCE'] = np.nanmean(mce_list)
    
    return calibration_data


def compute_dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Dict[str, Dict]:
    """
    Compute decision curve analysis (DCA) data for each class (one-vs-rest).
    
    Args:
        y_true: Ground-truth labels of shape (N,).
        y_prob: Predicted class probabilities of shape (N, 3).
        thresholds: Probability threshold array. Defaults to np.arange(0.01, 0.99, 0.01).
    
    Returns:
        Dict mapping class name to {thresholds, net_benefit, treat_all, treat_none}.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    
    n = len(y_true)
    dca_data = {}
    
    for i, cls_name in enumerate(CLASS_NAMES):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        prevalence = y_true_binary.mean()
        
        net_benefits = []
        
        for pt in thresholds:

            pred_positive = y_prob_class >= pt
            
            tp = np.sum((pred_positive == 1) & (y_true_binary == 1))
            fp = np.sum((pred_positive == 1) & (y_true_binary == 0))
            
            # Net Benefit = TP/N - FP/N * (pt / (1 - pt))
            if pt < 1:
                nb = tp / n - fp / n * (pt / (1 - pt))
            else:
                nb = 0
            
            net_benefits.append(nb)
        
        # Treat All baseline
        treat_all = []
        for pt in thresholds:
            if pt < 1:
                nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
            else:
                nb_all = 0
            treat_all.append(nb_all)
        
        dca_data[cls_name] = {
            'thresholds': thresholds,
            'net_benefit': np.array(net_benefits),
            'treat_all': np.array(treat_all),
            'treat_none': np.zeros_like(thresholds)
        }
    
    return dca_data


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Estimate metric confidence intervals via the percentile bootstrap method.
    
    Args:
        y_true: Ground-truth labels of shape (N,).
        y_prob: Predicted class probabilities of shape (N, 3).
        n_iterations : Number of bootstrap resamples.
        ci_level     : Nominal coverage probability (default: 0.95).
        random_state : NumPy random seed.
    
    Returns:
        Dict mapping metric name to {point, ci_lower, ci_upper, std}.
            {metric_name: {'point': float, 'ci_lower': float, 'ci_upper': float}}
    """
    np.random.seed(random_state)
    n = len(y_true)
    

    point_estimates = compute_all_metrics(y_true, y_prob)
    
    # Bootstrap
    bootstrap_results = {key: [] for key in point_estimates.keys()}
    
    for _ in range(n_iterations):

        indices = np.random.randint(0, n, size=n)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics_boot = compute_all_metrics(y_true_boot, y_prob_boot)
        
        for key, value in metrics_boot.items():
            if not np.isnan(value):
                bootstrap_results[key].append(value)
    

    alpha = 1 - ci_level
    results = {}
    
    for key, values in bootstrap_results.items():
        if len(values) > 0:
            values = np.array(values)
            ci_lower = np.percentile(values, alpha / 2 * 100)
            ci_upper = np.percentile(values, (1 - alpha / 2) * 100)
            results[key] = {
                'point': point_estimates[key],
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(values)
            }
        else:
            results[key] = {
                'point': point_estimates[key],
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'std': np.nan
            }
    
    return results


def format_metrics_report(
    metrics_with_ci: Dict[str, Dict[str, float]],
    precision: int = 4
) -> str:
    """
    Format a metric-with-CI dictionary as a human-readable report string.
    
    Args:
        metrics_with_ci: Return value of bootstrap_metrics().
        precision      : Decimal places for formatting.
    
    Returns:
        Formatted multi-line report string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Performance Metrics with 95% Confidence Intervals")
    lines.append("=" * 70)
    

    lines.append("\n[Global Metrics]")
    global_metrics = ['AUC_Macro', 'Accuracy', 'F1_Macro', 'Kappa', 'MCC', 'Brier_Score']
    for metric in global_metrics:
        if metric in metrics_with_ci:
            m = metrics_with_ci[metric]
            lines.append(
                f"  {metric}: {m['point']:.{precision}f} "
                f"(95% CI: {m['ci_lower']:.{precision}f}-{m['ci_upper']:.{precision}f})"
            )
    

    for cls_name in CLASS_NAMES:
        lines.append(f"\n[{cls_name} Metrics]")
        class_metrics = ['AUC', 'Sens', 'Spec', 'Prec', 'NPV', 'F1']
        for metric in class_metrics:
            key = f"{cls_name}_{metric}"
            if key in metrics_with_ci:
                m = metrics_with_ci[key]
                lines.append(
                    f"  {metric}: {m['point']:.{precision}f} "
                    f"(95% CI: {m['ci_lower']:.{precision}f}-{m['ci_upper']:.{precision}f})"
                )
    
    lines.append("\n" + "=" * 70)
    
    return '\n'.join(lines)

