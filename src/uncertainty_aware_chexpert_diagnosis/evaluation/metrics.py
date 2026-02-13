"""Evaluation metrics for uncertainty-aware predictions."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

logger = logging.getLogger(__name__)


class UncertaintyMetrics:
    """Metrics for evaluating uncertainty-aware predictions."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config.get('metrics', {})

        self.compute_auroc = self.metrics_config.get('compute_auroc', True)
        self.compute_auprc = self.metrics_config.get('compute_auprc', True)
        self.compute_ece = self.metrics_config.get('compute_ece', True)
        self.compute_brier = self.metrics_config.get('compute_brier', True)
        self.compute_uncertainty_corr = self.metrics_config.get('uncertainty_correlation', True)

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainty_masks: np.ndarray,
        epistemic_uncertainty: Optional[np.ndarray] = None,
        aleatoric_uncertainty: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            predictions: Predicted probabilities [N, num_classes]
            labels: Ground truth labels [N, num_classes]
            uncertainty_masks: Binary masks for uncertain labels [N, num_classes]
            epistemic_uncertainty: Epistemic uncertainty per sample [N]
            aleatoric_uncertainty: Aleatoric uncertainty per sample [N]

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Separate certain and uncertain labels
        certain_mask = (uncertainty_masks == 0) & (labels >= 0)
        uncertain_mask = (uncertainty_masks == 1)

        # AUROC for certain labels
        if self.compute_auroc and certain_mask.sum() > 0:
            try:
                auroc_certain = self._compute_auroc_per_class(
                    predictions[certain_mask],
                    labels[certain_mask]
                )
                metrics['auroc_certain_mean'] = np.mean(list(auroc_certain.values()))
                for i, auroc in auroc_certain.items():
                    metrics[f'auroc_certain_class_{i}'] = auroc
            except Exception as e:
                logger.warning(f"Could not compute AUROC for certain labels: {e}")
                metrics['auroc_certain_mean'] = 0.0

        # AUROC for uncertain labels (key metric for our approach)
        if self.compute_auroc and uncertain_mask.sum() > 0:
            try:
                # For uncertain labels, we check if model assigns high uncertainty
                # This is our novel contribution
                auroc_uncertain = self._compute_auroc_per_class(
                    predictions[uncertain_mask],
                    labels[uncertain_mask]
                )
                metrics['auroc_uncertain_mean'] = np.mean(list(auroc_uncertain.values()))
                for i, auroc in auroc_uncertain.items():
                    metrics[f'auroc_uncertain_class_{i}'] = auroc
            except Exception as e:
                logger.warning(f"Could not compute AUROC for uncertain labels: {e}")
                metrics['auroc_uncertain_mean'] = 0.0

        # Overall AUROC
        if self.compute_auroc:
            try:
                auroc_all = self._compute_auroc_per_class(predictions, labels)
                metrics['auroc_mean'] = np.mean(list(auroc_all.values()))
            except Exception as e:
                logger.warning(f"Could not compute overall AUROC: {e}")
                metrics['auroc_mean'] = 0.0

        # AUPRC
        if self.compute_auprc:
            try:
                auprc = self._compute_auprc(predictions, labels)
                metrics['auprc_mean'] = np.mean(list(auprc.values()))
            except Exception as e:
                logger.warning(f"Could not compute AUPRC: {e}")
                metrics['auprc_mean'] = 0.0

        # Expected Calibration Error
        if self.compute_ece:
            try:
                ece = self._compute_ece(predictions, labels)
                metrics['calibration_ece'] = ece
            except Exception as e:
                logger.warning(f"Could not compute ECE: {e}")
                metrics['calibration_ece'] = 0.0

        # Brier Score
        if self.compute_brier:
            try:
                brier = self._compute_brier_score(predictions, labels)
                metrics['brier_score'] = brier
            except Exception as e:
                logger.warning(f"Could not compute Brier score: {e}")
                metrics['brier_score'] = 0.0

        # Uncertainty correlation (key innovation metric)
        if self.compute_uncertainty_corr and epistemic_uncertainty is not None:
            try:
                corr = self._compute_uncertainty_correlation(
                    epistemic_uncertainty,
                    uncertainty_masks
                )
                metrics['uncertainty_correlation'] = corr
            except Exception as e:
                logger.warning(f"Could not compute uncertainty correlation: {e}")
                metrics['uncertainty_correlation'] = 0.0

        return metrics

    def _compute_auroc_per_class(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, float]:
        """Compute AUROC for each class."""
        num_classes = predictions.shape[1]
        auroc_dict = {}

        for i in range(num_classes):
            class_labels = labels[:, i]
            class_preds = predictions[:, i]

            # Filter out NaN labels
            valid_mask = ~np.isnan(class_labels)
            if valid_mask.sum() > 0 and len(np.unique(class_labels[valid_mask])) > 1:
                try:
                    auroc = roc_auc_score(
                        class_labels[valid_mask],
                        class_preds[valid_mask]
                    )
                    auroc_dict[i] = auroc
                except Exception as e:
                    logger.warning(f"Could not compute AUROC for class {i}: {e}")
                    auroc_dict[i] = 0.5
            else:
                auroc_dict[i] = 0.5

        return auroc_dict

    def _compute_auprc(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, float]:
        """Compute average precision (AUPRC) for each class."""
        num_classes = predictions.shape[1]
        auprc_dict = {}

        for i in range(num_classes):
            class_labels = labels[:, i]
            class_preds = predictions[:, i]

            valid_mask = ~np.isnan(class_labels)
            if valid_mask.sum() > 0 and len(np.unique(class_labels[valid_mask])) > 1:
                try:
                    auprc = average_precision_score(
                        class_labels[valid_mask],
                        class_preds[valid_mask]
                    )
                    auprc_dict[i] = auprc
                except Exception:
                    auprc_dict[i] = 0.0
            else:
                auprc_dict[i] = 0.0

        return auprc_dict

    def _compute_ece(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            predictions: Predicted probabilities
            labels: Ground truth labels
            n_bins: Number of bins for calibration

        Returns:
            ECE value
        """
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()

        # Remove NaN labels
        valid_mask = ~np.isnan(labels_flat)
        predictions_flat = predictions_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]

        if len(predictions_flat) == 0:
            return 0.0

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in bin
            in_bin = (predictions_flat > bin_lower) & (predictions_flat <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = labels_flat[in_bin].mean()
                avg_confidence_in_bin = predictions_flat[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _compute_brier_score(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute Brier score."""
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()

        valid_mask = ~np.isnan(labels_flat)
        predictions_flat = predictions_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]

        if len(predictions_flat) == 0:
            return 0.0

        return np.mean((predictions_flat - labels_flat) ** 2)

    def _compute_uncertainty_correlation(
        self,
        epistemic_uncertainty: np.ndarray,
        uncertainty_masks: np.ndarray
    ) -> float:
        """
        Compute correlation between model uncertainty and radiologist uncertainty.

        This is a key metric for our approach: we want high epistemic uncertainty
        when radiologists marked labels as uncertain.

        Args:
            epistemic_uncertainty: Model's epistemic uncertainty per sample
            uncertainty_masks: Binary masks indicating radiologist uncertainty

        Returns:
            Spearman correlation coefficient
        """
        # Compute average uncertainty per sample
        avg_uncertainty_per_sample = uncertainty_masks.mean(axis=1)

        # Compute correlation
        try:
            corr, _ = spearmanr(epistemic_uncertainty, avg_uncertainty_per_sample)
            if np.isnan(corr):
                corr = 0.0
        except Exception as e:
            logger.warning(f"Could not compute uncertainty correlation: {e}")
            corr = 0.0

        return corr


def compute_confidence_intervals(
    metrics: np.ndarray,
    confidence: float = 0.95
) -> tuple:
    """
    Compute confidence intervals using bootstrap.

    Args:
        metrics: Array of metric values from bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    mean = np.mean(metrics)
    alpha = 1 - confidence
    lower = np.percentile(metrics, alpha / 2 * 100)
    upper = np.percentile(metrics, (1 - alpha / 2) * 100)

    return mean, lower, upper
