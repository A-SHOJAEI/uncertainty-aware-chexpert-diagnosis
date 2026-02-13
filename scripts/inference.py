#!/usr/bin/env python
"""Inference script for uncertainty-aware CheXpert diagnosis with automated referral."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from uncertainty_aware_chexpert_diagnosis.data.preprocessing import get_transforms
from uncertainty_aware_chexpert_diagnosis.models.model import EvidentialCheXpertModel
from uncertainty_aware_chexpert_diagnosis.utils.config import (
    load_config,
    setup_logging,
    get_device,
)

logger = logging.getLogger(__name__)


class CheXpertInference:
    """
    Inference class for CheXpert diagnosis with uncertainty-based referral.

    Attributes:
        model: Trained model
        device: PyTorch device
        transforms: Image preprocessing transforms
        config: Configuration dictionary
        class_names: List of disease class names
        uncertainty_threshold: Threshold for automated referral
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        uncertainty_threshold: float = 0.7,
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file
            uncertainty_threshold: Epistemic uncertainty threshold for referral
        """
        self.device = get_device()
        self.config = load_config(config_path)
        self.uncertainty_threshold = uncertainty_threshold

        # Load class names
        self.class_names = self.config.get('data', {}).get('labels', [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion"
        ])

        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            model_config = self.config

        self.model = EvidentialCheXpertModel(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get transforms
        self.transforms = get_transforms(self.config, train=False)

        logger.info("Inference engine initialized")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor [1, 3, H, W]

        Raises:
            FileNotFoundError: If image file does not exist
            ValueError: If image cannot be loaded
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Load image
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Apply transforms
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented['image']

            # Add batch dimension
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()

            if image.ndim == 3:
                image = image.unsqueeze(0)

            return image

        except Exception as e:
            raise ValueError(f"Error preprocessing image {image_path}: {e}")

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        Run inference on a single image.

        Args:
            image_path: Path to chest X-ray image

        Returns:
            Dictionary containing:
                - predictions: Disease probabilities per class
                - epistemic: Epistemic uncertainty per class
                - aleatoric: Aleatoric uncertainty per class
                - referral_needed: Boolean indicating if referral is recommended
                - referral_reason: String explaining referral reason
                - confidence: Overall prediction confidence
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Forward pass
        outputs = self.model(image_tensor)

        # Extract predictions
        probs = outputs['prob'].cpu().numpy()[0]
        epistemic = outputs.get('epistemic', torch.zeros(1)).cpu().numpy()[0]

        # Per-class results
        predictions = []
        for i, class_name in enumerate(self.class_names):
            predictions.append({
                'class': class_name,
                'probability': float(probs[i]),
                'epistemic_uncertainty': float(epistemic) if isinstance(epistemic, (int, float)) else 0.0,
            })

        # Determine referral need
        mean_epistemic = float(epistemic) if isinstance(epistemic, (int, float)) else 0.0
        referral_needed = mean_epistemic > self.uncertainty_threshold

        if referral_needed:
            referral_reason = f"High epistemic uncertainty ({mean_epistemic:.3f} > {self.uncertainty_threshold}). Model lacks sufficient confidence - recommend specialist review."
        else:
            referral_reason = "Sufficient model confidence for automated triage."

        # Calculate overall confidence
        confidence = 1.0 - mean_epistemic

        result = {
            'image_path': str(image_path),
            'predictions': predictions,
            'epistemic_uncertainty': mean_epistemic,
            'referral_needed': referral_needed,
            'referral_reason': referral_reason,
            'confidence': confidence,
        }

        return result

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of paths to chest X-ray images

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'referral_needed': True,
                    'referral_reason': 'Processing error - manual review required',
                })

        return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on chest X-ray images with automated referral"
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
        "--image",
        type=str,
        help="Path to single image"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/inference_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.7,
        help="Epistemic uncertainty threshold for referral (default: 0.7)"
    )

    return parser.parse_args()


def main() -> None:
    """Main inference function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting inference")

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = Path(__file__).parent.parent / args.checkpoint

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please train the model first using: python scripts/train.py")
        return

    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config

    # Initialize inference engine
    try:
        inference_engine = CheXpertInference(
            str(checkpoint_path),
            str(config_path),
            args.uncertainty_threshold
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        return

    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            logger.error(f"Image directory not found: {image_dir}")
            return
        image_paths.extend([
            str(p) for p in image_dir.glob("*.jpg")
        ])
        image_paths.extend([
            str(p) for p in image_dir.glob("*.png")
        ])
    else:
        logger.error("Please specify --image or --image-dir")
        return

    if not image_paths:
        logger.error("No images found")
        return

    logger.info(f"Processing {len(image_paths)} images")

    # Run inference
    results = inference_engine.predict_batch(image_paths)

    # Print results
    logger.info("\n" + "="*50)
    logger.info("Inference Results")
    logger.info("="*50)

    referral_count = 0
    for result in results:
        if 'error' in result:
            logger.error(f"\n{result['image_path']}: ERROR - {result['error']}")
            continue

        logger.info(f"\n{result['image_path']}:")
        logger.info(f"  Confidence: {result['confidence']:.3f}")
        logger.info(f"  Epistemic Uncertainty: {result['epistemic_uncertainty']:.3f}")
        logger.info(f"  Referral Needed: {result['referral_needed']}")

        if result['referral_needed']:
            referral_count += 1
            logger.info(f"  Reason: {result['referral_reason']}")

        logger.info("  Predictions:")
        for pred in result['predictions']:
            logger.info(f"    {pred['class']}: {pred['probability']:.3f}")

    logger.info("\n" + "="*50)
    logger.info(f"Total Images: {len(results)}")
    logger.info(f"Referrals Recommended: {referral_count} ({100*referral_count/len(results):.1f}%)")
    logger.info(f"Automated Triage: {len(results) - referral_count} ({100*(len(results) - referral_count)/len(results):.1f}%)")
    logger.info("="*50)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
