"""Tests for training utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from uncertainty_aware_chexpert_diagnosis.models.model import EvidentialCheXpertModel
from uncertainty_aware_chexpert_diagnosis.training.trainer import EvidentialTrainer
from uncertainty_aware_chexpert_diagnosis.evaluation.metrics import UncertaintyMetrics


def create_dummy_dataloader(batch_size=4, num_batches=5, num_classes=5, image_size=224):
    """Create a dummy dataloader for testing."""
    num_samples = batch_size * num_batches

    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 2, (num_samples, num_classes)).float()
    uncertainty_masks = torch.randint(0, 2, (num_samples, num_classes)).float()

    # Create dataset that returns dictionaries
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, uncertainty_masks):
            self.images = images
            self.labels = labels
            self.uncertainty_masks = uncertainty_masks

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return {
                'image': self.images[idx],
                'label': self.labels[idx],
                'uncertainty_mask': self.uncertainty_masks[idx]
            }

    dataset = DictDataset(images, labels, uncertainty_masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_trainer_initialization(config, device):
    """Test trainer initialization."""
    model = EvidentialCheXpertModel(config)
    trainer = EvidentialTrainer(model, config, device)

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.epochs == config['training']['epochs']


def test_trainer_create_optimizer(config, device):
    """Test optimizer creation."""
    model = EvidentialCheXpertModel(config)

    # Test Adam
    config['training']['optimizer'] = 'adam'
    trainer = EvidentialTrainer(model, config, device)
    assert isinstance(trainer.optimizer, torch.optim.Adam)

    # Test AdamW
    config['training']['optimizer'] = 'adamw'
    trainer = EvidentialTrainer(model, config, device)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)

    # Test SGD
    config['training']['optimizer'] = 'sgd'
    trainer = EvidentialTrainer(model, config, device)
    assert isinstance(trainer.optimizer, torch.optim.SGD)


def test_trainer_create_scheduler(config, device):
    """Test scheduler creation."""
    model = EvidentialCheXpertModel(config)

    # Test cosine scheduler
    config['training']['scheduler'] = 'cosine'
    trainer = EvidentialTrainer(model, config, device)
    assert trainer.scheduler is not None

    # Test step scheduler
    config['training']['scheduler'] = 'step'
    trainer = EvidentialTrainer(model, config, device)
    assert trainer.scheduler is not None

    # Test no scheduler
    config['training']['scheduler'] = 'none'
    trainer = EvidentialTrainer(model, config, device)
    assert trainer.scheduler is None


def test_trainer_compute_loss(config, device, sample_batch):
    """Test loss computation."""
    model = EvidentialCheXpertModel(config)
    trainer = EvidentialTrainer(model, config, device)

    # Move batch to device
    for key in sample_batch:
        sample_batch[key] = sample_batch[key].to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(sample_batch['image'])

    loss, loss_dict = trainer._compute_loss(
        outputs,
        sample_batch['label'],
        sample_batch['uncertainty_mask'],
        epoch=0
    )

    assert loss.item() >= 0
    assert 'cls_loss' in loss_dict
    assert 'unc_loss' in loss_dict


def test_trainer_train_epoch(config, device):
    """Test training for one epoch."""
    # Simplify config for faster testing
    config['training']['batch_size'] = 4
    config['training']['use_amp'] = False

    model = EvidentialCheXpertModel(config)
    trainer = EvidentialTrainer(model, config, device)

    # Create dummy dataloader
    train_loader = create_dummy_dataloader(batch_size=4, num_batches=2)

    # Train one epoch
    metrics = trainer.train_epoch(train_loader, epoch=1)

    assert 'train_loss' in metrics
    assert 'train_cls_loss' in metrics
    assert 'train_unc_loss' in metrics
    assert metrics['train_loss'] >= 0


def test_trainer_validate(config, device):
    """Test validation."""
    config['training']['batch_size'] = 4

    model = EvidentialCheXpertModel(config)
    trainer = EvidentialTrainer(model, config, device)

    # Create dummy dataloader
    val_loader = create_dummy_dataloader(batch_size=4, num_batches=2)

    # Validate
    metrics = trainer.validate(val_loader, epoch=1)

    assert 'val_loss' in metrics
    assert 'auroc_mean' in metrics
    assert metrics['val_loss'] >= 0


def test_trainer_checkpoint_save_load(config, device, tmp_path):
    """Test checkpoint saving and loading."""
    config['experiment']['checkpoint_dir'] = str(tmp_path)

    model = EvidentialCheXpertModel(config)
    trainer = EvidentialTrainer(model, config, device)

    # Save checkpoint
    metrics = {'auroc_mean': 0.85}
    trainer.save_checkpoint(epoch=1, metrics=metrics, is_best=True)

    checkpoint_path = tmp_path / 'best_model.pth'
    assert checkpoint_path.exists()

    # Load checkpoint
    new_model = EvidentialCheXpertModel(config)
    new_trainer = EvidentialTrainer(new_model, config, device)
    new_trainer.load_checkpoint(str(checkpoint_path))


def test_uncertainty_metrics_initialization(config):
    """Test metrics initialization."""
    metrics = UncertaintyMetrics(config)

    assert metrics.compute_auroc
    assert metrics.compute_ece


def test_uncertainty_metrics_compute(config, sample_predictions):
    """Test metrics computation."""
    metrics_calculator = UncertaintyMetrics(config)

    metrics = metrics_calculator.compute_metrics(
        sample_predictions['predictions'],
        sample_predictions['labels'],
        sample_predictions['uncertainty_masks'],
        sample_predictions['epistemic'],
        sample_predictions['aleatoric']
    )

    assert 'auroc_mean' in metrics
    assert 'calibration_ece' in metrics
    assert 'brier_score' in metrics
    assert 'uncertainty_correlation' in metrics


def test_uncertainty_metrics_auroc(config, sample_predictions):
    """Test AUROC computation."""
    metrics_calculator = UncertaintyMetrics(config)

    auroc_dict = metrics_calculator._compute_auroc_per_class(
        sample_predictions['predictions'],
        sample_predictions['labels']
    )

    assert len(auroc_dict) == sample_predictions['predictions'].shape[1]

    for class_id, auroc in auroc_dict.items():
        assert 0 <= auroc <= 1


def test_uncertainty_metrics_ece(config, sample_predictions):
    """Test ECE computation."""
    metrics_calculator = UncertaintyMetrics(config)

    ece = metrics_calculator._compute_ece(
        sample_predictions['predictions'],
        sample_predictions['labels']
    )

    assert 0 <= ece <= 1


def test_uncertainty_metrics_brier(config, sample_predictions):
    """Test Brier score computation."""
    metrics_calculator = UncertaintyMetrics(config)

    brier = metrics_calculator._compute_brier_score(
        sample_predictions['predictions'],
        sample_predictions['labels']
    )

    assert 0 <= brier <= 1


def test_uncertainty_metrics_correlation(config, sample_predictions):
    """Test uncertainty correlation computation."""
    metrics_calculator = UncertaintyMetrics(config)

    corr = metrics_calculator._compute_uncertainty_correlation(
        sample_predictions['epistemic'],
        sample_predictions['uncertainty_masks']
    )

    assert -1 <= corr <= 1
