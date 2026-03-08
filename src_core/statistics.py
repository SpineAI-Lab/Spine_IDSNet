"""
SpineIDS Clinical ML - Statistical Analysis
====================================
Contents:
- Bootstrap confidence intervals
- DeLong test for paired AUC comparison
- Multi-model statistical comparison
- Summary reporting
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

from .config import (
    BOOTSTRAP_N_ITERATIONS, BOOTSTRAP_CI_LEVEL,
    CLASS_NAMES, OUTPUT_DIR
)
from .metrics import (
    compute_all_metrics, bootstrap_metrics, format_metrics_report
)
from .utils import (
    setup_logging, save_json, print_section_header, print_subsection_header
)


def delong_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray
) -> Tuple[float, float]:
    """
    DeLong test for the difference between two correlated AUCs.

    Reference: DeLong et al. (1988). Comparing the areas under two or more
    correlated receiver operating characteristic curves: a nonparametric approach.
    
    Reference: DeLong et al. (1988). Comparing the areas under two or more 
    correlated receiver operating characteristic curves
    
    Args:
        y_true: Binary ground-truth labels (0 / 1).
        prob_a: Predicted probabilities for model A.
        prob_b: Predicted probabilities for model B.
    
    Returns:
        z_stat : Z-statistic.
        p_value: Two-sided p-value.
    """

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    

    def compute_placement_values(probs, pos_idx, neg_idx):
        placements = np.zeros(len(pos_idx))
        for i, p in enumerate(pos_idx):
            count = np.sum(probs[p] > probs[neg_idx])
            ties = np.sum(probs[p] == probs[neg_idx])
            placements[i] = (count + 0.5 * ties) / len(neg_idx)
        return placements
    

    pv_a = compute_placement_values(prob_a, pos_idx, neg_idx)
    pv_b = compute_placement_values(prob_b, pos_idx, neg_idx)
    
    # AUC
    auc_a = np.mean(pv_a)
    auc_b = np.mean(pv_b)
    

    var_a = np.var(pv_a, ddof=1) / n_pos
    var_b = np.var(pv_b, ddof=1) / n_pos
    cov_ab = np.cov(pv_a, pv_b, ddof=1)[0, 1] / n_pos
    

    def compute_neg_placement_values(probs, pos_idx, neg_idx):
        placements = np.zeros(len(neg_idx))
        for i, n in enumerate(neg_idx):
            count = np.sum(probs[n] < probs[pos_idx])
            ties = np.sum(probs[n] == probs[pos_idx])
            placements[i] = (count + 0.5 * ties) / len(pos_idx)
        return placements
    
    npv_a = compute_neg_placement_values(prob_a, pos_idx, neg_idx)
    npv_b = compute_neg_placement_values(prob_b, pos_idx, neg_idx)
    
    var_a_neg = np.var(npv_a, ddof=1) / n_neg
    var_b_neg = np.var(npv_b, ddof=1) / n_neg
    cov_ab_neg = np.cov(npv_a, npv_b, ddof=1)[0, 1] / n_neg
    

    var_diff = var_a + var_b - 2 * cov_ab + var_a_neg + var_b_neg - 2 * cov_ab_neg
    
    if var_diff <= 0:
        return np.nan, np.nan
    

    z_stat = (auc_a - auc_b) / np.sqrt(var_diff)
    

    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value


def compare_models_delong(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    class_idx: int = 0
) -> pd.DataFrame:
    """
    Compare multiple classifiers via pairwise DeLong tests.
    
    Args:
        y_true: Multi-class ground-truth labels.
        model_probs : Dict mapping model name to (N, 3) probability array.
        class_idx   : Index of the class used for OvR AUC computation.
    
    Returns:
        Symmetric p-value DataFrame indexed by model name.
    """
    model_names = list(model_probs.keys())
    n_models = len(model_names)
    

    y_binary = (y_true == class_idx).astype(int)
    

    pvalue_matrix = np.ones((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            prob_i = model_probs[model_names[i]][:, class_idx]
            prob_j = model_probs[model_names[j]][:, class_idx]
            
            try:
                _, p_value = delong_test(y_binary, prob_i, prob_j)
                pvalue_matrix[i, j] = p_value
                pvalue_matrix[j, i] = p_value
            except:
                pvalue_matrix[i, j] = np.nan
                pvalue_matrix[j, i] = np.nan
    

    pvalue_df = pd.DataFrame(
        pvalue_matrix,
        index=model_names,
        columns=model_names
    )
    
    return pvalue_df


def compare_models_wilcoxon(
    fold_metrics: Dict[str, List[Dict[str, float]]],
    metric_name: str = 'AUC_Macro'
) -> pd.DataFrame:
    """
    Compare classifiers using the Wilcoxon signed-rank test on per-fold metric values.
    
    Args:
        fold_metrics: {model_name: [fold_metrics_dict, ...]}
        metric_name: Key of the metric to compare (default: ``AUC_Macro``).
    
    Returns:
        Symmetric p-value DataFrame indexed by model name.
    """
    model_names = list(fold_metrics.keys())
    n_models = len(model_names)
    

    model_values = {}
    for model_name, fold_results in fold_metrics.items():
        model_values[model_name] = [f[metric_name] for f in fold_results]
    

    pvalue_matrix = np.ones((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            values_i = model_values[model_names[i]]
            values_j = model_values[model_names[j]]
            
            try:

                _, p_value = stats.wilcoxon(values_i, values_j)
                pvalue_matrix[i, j] = p_value
                pvalue_matrix[j, i] = p_value
            except:
                pvalue_matrix[i, j] = np.nan
                pvalue_matrix[j, i] = np.nan
    

    pvalue_df = pd.DataFrame(
        pvalue_matrix,
        index=model_names,
        columns=model_names
    )
    
    return pvalue_df


def bonferroni_correction(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Bonferroni correction and return a boolean significance mask."""
    n_comparisons = np.sum(~np.isnan(pvalues)) // 2
    adjusted_alpha = alpha / max(n_comparisons, 1)
    return pvalues < adjusted_alpha


class StatisticalAnalyzer:
    """StatisticalAnalyzer"""
    
    def __init__(
        self,
        n_bootstrap: int = BOOTSTRAP_N_ITERATIONS,
        ci_level: float = BOOTSTRAP_CI_LEVEL,
        logger=None
    ):
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.logger = logger or setup_logging()
    
    def analyze_single_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Compute bootstrap CIs for a single model and optionally persist results.
        
        Returns:
            Dict of metric name -> {point, ci_lower, ci_upper, std}.
        """
        print_subsection_header("Computing Bootstrap CI")
        

        metrics_with_ci = bootstrap_metrics(
            y_true, y_prob,
            n_iterations=self.n_bootstrap,
            ci_level=self.ci_level
        )
        

        report = format_metrics_report(metrics_with_ci)
        self.logger.info(f"\n{report}")
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            

            results_df = pd.DataFrame(metrics_with_ci).T
            results_df.to_csv(output_dir / 'metrics_with_95CI.csv')
            

            with open(output_dir / 'metrics_with_95CI.txt', 'w') as f:
                f.write(report)
        
        return metrics_with_ci
    
    def compare_models(
        self,
        y_true: np.ndarray,
        model_probs: Dict[str, np.ndarray],
        fold_metrics: Optional[Dict[str, List[Dict]]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Comprehensive multi-model comparison (metrics + DeLong + optional Wilcoxon).
        
        Args:
            y_true      : Ground-truth labels.
            model_probs : Dict mapping model name to (N, 3) probability array.
            fold_metrics: Per-fold metric dicts, used for Wilcoxon test (optional).
            output_dir: output_dir  : Directory for saving comparison results (optional).
        
        Returns:
            Dict containing metrics_summary, delong_pvalues, wilcoxon_pvalues.
        """
        print_section_header("Model Comparison")
        
        results = {}
        

        self.logger.info("Computing metrics for each model...")
        all_metrics = {}
        for model_name, probs in model_probs.items():
            metrics = compute_all_metrics(y_true, probs)
            all_metrics[model_name] = metrics
        

        metrics_df = pd.DataFrame(all_metrics).T
        results['metrics_summary'] = metrics_df
        
        self.logger.info(f"\nMetrics Summary:")
        self.logger.info(f"\n{metrics_df[['AUC_Macro', 'Accuracy', 'F1_Macro']].round(4)}")
        

        self.logger.info("\nRunning DeLong tests for AUC comparison...")
        delong_results = {}
        
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            pvalue_df = compare_models_delong(y_true, model_probs, cls_idx)
            delong_results[cls_name] = pvalue_df
        


        macro_pvalues = np.maximum.reduce([
            delong_results[cls].values for cls in CLASS_NAMES
        ])
        macro_pvalue_df = pd.DataFrame(
            macro_pvalues,
            index=list(model_probs.keys()),
            columns=list(model_probs.keys())
        )
        results['delong_pvalues'] = macro_pvalue_df
        
        self.logger.info(f"\nDeLong p-values (Macro):")
        self.logger.info(f"\n{macro_pvalue_df.round(4)}")
        

        if fold_metrics is not None:
            self.logger.info("\nRunning Wilcoxon tests...")
            wilcoxon_df = compare_models_wilcoxon(fold_metrics, 'AUC_Macro')
            results['wilcoxon_pvalues'] = wilcoxon_df
            
            self.logger.info(f"\nWilcoxon p-values:")
            self.logger.info(f"\n{wilcoxon_df.round(4)}")
        

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_df.to_csv(output_dir / 'all_models_metrics.csv')
            macro_pvalue_df.to_csv(output_dir / 'delong_pvalues.csv')
            
            if 'wilcoxon_pvalues' in results:
                results['wilcoxon_pvalues'].to_csv(output_dir / 'wilcoxon_pvalues.csv')
            
            self.logger.info(f"\nResults saved to {output_dir}")
        
        return results
    
    def select_best_model(
        self,
        metrics_summary: pd.DataFrame,
        primary_metric: str = 'AUC_Macro',
        secondary_metrics: List[str] = ['Accuracy', 'F1_Macro']
    ) -> str:
        """
        Select the best model from a metrics summary DataFrame.
        
        Args:
            metrics_summary  : DataFrame with model names as index.
            primary_metric   : Column name for primary ranking criterion.
            secondary_metrics: Additional metrics to log for the top model.
        
        Returns:
            Name of the best-performing model.
        """

        sorted_df = metrics_summary.sort_values(primary_metric, ascending=False)
        
        best_model = sorted_df.index[0]
        best_score = sorted_df.loc[best_model, primary_metric]
        
        self.logger.info(f"\nBest model by {primary_metric}: {best_model}")
        self.logger.info(f"  {primary_metric}: {best_score:.4f}")
        
        for metric in secondary_metrics:
            score = sorted_df.loc[best_model, metric]
            self.logger.info(f"  {metric}: {score:.4f}")
        
        return best_model

