#!/usr/bin/env python
"""Simple prediction script for uncertainty-aware CheXpert diagnosis.

This script provides a straightforward interface for running inference on
chest X-ray images using a trained model checkpoint.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np

from uncertainty_aware_chexpert_diagnosis.data.preprocessing import get_transforms
from uncertainty_aware_chexpert_diagnosis.models.model import EvidentialCheXpertModel
from uncertainty_aware_chexpert_diagnosis.utils.config import (
    load_config,
    setup_logging,
    get_device,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run prediction on a chest X-ray image"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to chest X-ray image"
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
        help="Path to save prediction results (optional)"
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: PyTorch device

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = config

    model = EvidentialCheXpertModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image_path: str, transforms) -> torch.Tensor:
    """
    Preprocess image for model input.

    Args:
        image_path: Path to image file
        transforms: Preprocessing transforms

    Returns:
        Preprocessed image tensor [1, 3, H, W]
    """
    import cv2

    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Apply transforms
    if transforms:
        augmented = transforms(image=image)
        image = augmented['image']

    # Add batch dimension
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    if image.ndim == 3:
        image = image.unsqueeze(0)

    return image


@torch.no_grad()
def predict(model, image_tensor, device, class_names):
    """
    Run model prediction.

    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: PyTorch device
        class_names: List of class names

    Returns:
        Dictionary of predictions
    """
    image_tensor = image_tensor.to(device)

    # Forward pass
    outputs = model(image_tensor)

    # Extract predictions
    probs = outputs['prob'].cpu().numpy()[0]
    epistemic = outputs.get('epistemic', torch.zeros(1)).cpu().numpy()[0]
    aleatoric = outputs.get('aleatoric', torch.zeros(1)).cpu().numpy()[0]

    # Build results
    results = {
        'predictions': {}
    }

    for i, class_name in enumerate(class_names):
        results['predictions'][class_name] = {
            'probability': float(probs[i]),
            'prediction': 'positive' if probs[i] > 0.5 else 'negative'
        }

    # Add uncertainty metrics if available
    if isinstance(epistemic, np.ndarray):
        results['epistemic_uncertainty'] = float(epistemic.mean())
    else:
        results['epistemic_uncertainty'] = float(epistemic)

    if isinstance(aleatoric, np.ndarray):
        results['aleatoric_uncertainty'] = float(aleatoric.mean())
    else:
        results['aleatoric_uncertainty'] = float(aleatoric)

    return results


def main():
    """Main prediction function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config

    config = load_config(str(config_path))

    # Get class names
    class_names = config.get('data', {}).get('labels', [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion"
    ])

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

    # Load model
    model = load_model(str(checkpoint_path), config, device)

    # Get transforms
    transforms = get_transforms(config, train=False)

    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return

    # Preprocess image
    logger.info(f"Processing image: {image_path}")
    image_tensor = preprocess_image(str(image_path), transforms)

    # Run prediction
    results = predict(model, image_tensor, device, class_names)

    # Print results
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)

    for class_name, pred in results['predictions'].items():
        print(f"\n{class_name}:")
        print(f"  Probability: {pred['probability']:.3f}")
        print(f"  Prediction: {pred['prediction']}")

    print(f"\nEpistemic Uncertainty: {results['epistemic_uncertainty']:.3f}")
    print(f"Aleatoric Uncertainty: {results['aleatoric_uncertainty']:.3f}")
    print("="*50 + "\n")

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
