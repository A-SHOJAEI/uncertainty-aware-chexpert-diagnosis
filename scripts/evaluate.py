#!/usr/bin/env python
"""Evaluation script for uncertainty-aware CheXpert diagnosis."""

import argparse
from pathlib import Path
import json
import logging

import numpy as np
import torch
from tqdm import tqdm

from uncertainty_aware_chexpert_diagnosis.data.loader import CheXpertDataModule
from uncertainty_aware_chexpert_diagnosis.models.model import EvidentialCheXpertModel
from uncertainty_aware_chexpert_diagnosis.evaluation.metrics import UncertaintyMetrics
from uncertainty_aware_chexpert_diagnosis.utils.config import (
    load_config,
    setup_logging,
    get_device,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty-aware CheXpert model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metrics_calculator: UncertaintyMetrics,
    device: torch.device
) -> dict:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        metrics_calculator: Metrics calculator
        device: Device to use

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_uncertainty_masks = []
    all_epistemic = []
    all_aleatoric = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image'].to(device)
        labels = batch['label']
        uncertainty_mask = batch['uncertainty_mask']

        # Forward pass
        outputs = model(images)

        # Collect predictions
        all_probs.append(outputs['prob'].cpu().numpy())
        all_labels.append(labels.numpy())
        all_uncertainty_masks.append(uncertainty_mask.numpy())

        if 'epistemic' in outputs:
            all_epistemic.append(outputs['epistemic'].cpu().numpy())
        if 'aleatoric' in outputs:
            all_aleatoric.append(outputs['aleatoric'].cpu().numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_uncertainty_masks = np.concatenate(all_uncertainty_masks, axis=0)

    if all_epistemic:
        all_epistemic = np.concatenate(all_epistemic, axis=0)
    else:
        all_epistemic = None

    if all_aleatoric:
        all_aleatoric = np.concatenate(all_aleatoric, axis=0)
    else:
        all_aleatoric = None

    # Compute metrics
    metrics = metrics_calculator.compute_metrics(
        all_probs,
        all_labels,
        all_uncertainty_masks,
        all_epistemic,
        all_aleatoric
    )

    return metrics


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting evaluation")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config

    config = load_config(str(config_path))

    # Override batch size if specified
    if args.batch_size:
        config['inference'] = config.get('inference', {})
        config['inference']['batch_size'] = args.batch_size

    # Get device
    device = get_device()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = Path(__file__).parent.parent / args.checkpoint

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please train the model first using: python scripts/train.py")
        return

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = config

    model = EvidentialCheXpertModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    # Initialize data module
    data_module = CheXpertDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    # Get validation dataloader
    val_loader = data_module.val_dataloader()

    # Initialize metrics
    metrics_calculator = UncertaintyMetrics(config)

    # Evaluate
    logger.info("Running evaluation")
    metrics = evaluate_model(model, val_loader, metrics_calculator, device)

    # Print results
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results")
    logger.info("="*50)

    for metric_name, metric_value in sorted(metrics.items()):
        logger.info(f"{metric_name}: {metric_value:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Compare to target metrics
    target_metrics = config.get('metrics', {})
    logger.info("\n" + "="*50)
    logger.info("Comparison to Target Metrics")
    logger.info("="*50)

    comparisons = {
        'auroc_certain_mean': target_metrics.get('target_auroc_certain', 0.88),
        'auroc_uncertain_mean': target_metrics.get('target_auroc_uncertain', 0.75),
        'calibration_ece': target_metrics.get('target_ece', 0.05),
        'uncertainty_correlation': target_metrics.get('target_uncertainty_correlation', 0.7),
    }

    for metric_name, target_value in comparisons.items():
        if metric_name in metrics:
            actual_value = metrics[metric_name]
            diff = actual_value - target_value
            status = "✓" if (metric_name == 'calibration_ece' and diff < 0) or (metric_name != 'calibration_ece' and diff > 0) else "✗"
            logger.info(f"{status} {metric_name}: {actual_value:.4f} (target: {target_value:.4f}, diff: {diff:+.4f})")


if __name__ == "__main__":
    main()
