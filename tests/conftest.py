"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import numpy as np
import yaml


@pytest.fixture
def config():
    """Load default configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_batch():
    """Create a sample batch of data."""
    batch_size = 4
    num_classes = 5
    image_size = 224

    return {
        'image': torch.randn(batch_size, 3, image_size, image_size),
        'label': torch.randint(0, 2, (batch_size, num_classes)).float(),
        'uncertainty_mask': torch.randint(0, 2, (batch_size, num_classes)).float()
    }


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    num_samples = 100
    num_classes = 5

    predictions = np.random.rand(num_samples, num_classes)
    labels = np.random.randint(0, 2, (num_samples, num_classes)).astype(np.float32)
    uncertainty_masks = np.random.randint(0, 2, (num_samples, num_classes)).astype(np.float32)
    epistemic = np.random.rand(num_samples)
    aleatoric = np.random.rand(num_samples)

    return {
        'predictions': predictions,
        'labels': labels,
        'uncertainty_masks': uncertainty_masks,
        'epistemic': epistemic,
        'aleatoric': aleatoric
    }
