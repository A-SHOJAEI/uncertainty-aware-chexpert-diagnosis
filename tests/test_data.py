"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import torch

from uncertainty_aware_chexpert_diagnosis.data.preprocessing import (
    get_transforms,
    process_chexpert_labels,
    convert_to_rgb,
)
from uncertainty_aware_chexpert_diagnosis.data.loader import CheXpertDataModule


def test_get_transforms_training():
    """Test training transforms."""
    transforms = get_transforms(image_size=224, is_training=True)
    assert transforms is not None

    # Test on dummy image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    augmented = transforms(image=image)

    assert 'image' in augmented
    assert isinstance(augmented['image'], torch.Tensor)
    assert augmented['image'].shape == (3, 224, 224)


def test_get_transforms_validation():
    """Test validation transforms."""
    transforms = get_transforms(image_size=224, is_training=False)
    assert transforms is not None

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    augmented = transforms(image=image)

    assert 'image' in augmented
    assert augmented['image'].shape == (3, 224, 224)


def test_process_chexpert_labels_as_target():
    """Test label processing with uncertain as target policy."""
    labels = np.array([
        [1.0, 0.0, -1.0, np.nan, 1.0],
        [0.0, -1.0, 1.0, 0.0, -1.0]
    ])

    processed, uncertainty_mask = process_chexpert_labels(labels, uncertain_policy='as_target')

    assert processed.shape == labels.shape
    assert uncertainty_mask.shape == labels.shape
    assert uncertainty_mask.sum() == 3  # Three -1 values
    assert not np.isnan(processed).any()


def test_process_chexpert_labels_zeros():
    """Test label processing with U-Zeros policy."""
    labels = np.array([
        [1.0, 0.0, -1.0, np.nan, 1.0],
    ])

    processed, uncertainty_mask = process_chexpert_labels(labels, uncertain_policy='zeros')

    assert processed[0, 2] == 0.0  # -1 mapped to 0
    assert uncertainty_mask[0, 2] == 1.0


def test_process_chexpert_labels_ones():
    """Test label processing with U-Ones policy."""
    labels = np.array([
        [1.0, 0.0, -1.0, np.nan, 1.0],
    ])

    processed, uncertainty_mask = process_chexpert_labels(labels, uncertain_policy='ones')

    assert processed[0, 2] == 1.0  # -1 mapped to 1
    assert uncertainty_mask[0, 2] == 1.0


def test_convert_to_rgb():
    """Test grayscale to RGB conversion."""
    # Test 2D grayscale
    gray_2d = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    rgb = convert_to_rgb(gray_2d)
    assert rgb.shape == (256, 256, 3)

    # Test 3D grayscale with channel dim
    gray_3d = np.random.randint(0, 255, (256, 256, 1), dtype=np.uint8)
    rgb = convert_to_rgb(gray_3d)
    assert rgb.shape == (256, 256, 3)

    # Test already RGB
    rgb_input = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    rgb = convert_to_rgb(rgb_input)
    assert rgb.shape == (256, 256, 3)


def test_data_module_initialization(config):
    """Test data module initialization."""
    data_module = CheXpertDataModule(config)

    assert data_module.image_size == config['data']['image_size']
    assert data_module.uncertain_policy == config['data']['uncertain_policy']


def test_data_module_synthetic_dataset(config, tmp_path):
    """Test synthetic dataset creation."""
    # Use temporary directory
    config['data']['data_dir'] = str(tmp_path / 'test_data')

    data_module = CheXpertDataModule(config)
    data_module.prepare_data()

    # Check that directory was created
    data_dir = tmp_path / 'test_data'
    assert data_dir.exists()
    assert (data_dir / 'labels.csv').exists()


def test_data_module_setup(config, tmp_path):
    """Test data module setup."""
    config['data']['data_dir'] = str(tmp_path / 'test_data')

    data_module = CheXpertDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert len(data_module.train_dataset) > 0
    assert len(data_module.val_dataset) > 0


def test_data_module_dataloaders(config, tmp_path):
    """Test dataloader creation."""
    config['data']['data_dir'] = str(tmp_path / 'test_data')

    data_module = CheXpertDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    assert train_loader is not None
    assert val_loader is not None

    # Test one batch
    batch = next(iter(train_loader))
    assert 'image' in batch
    assert 'label' in batch
    assert 'uncertainty_mask' in batch

    batch_size = config['training']['batch_size']
    assert batch['image'].shape[0] == batch_size
    assert batch['label'].shape[0] == batch_size
    assert batch['uncertainty_mask'].shape[0] == batch_size
