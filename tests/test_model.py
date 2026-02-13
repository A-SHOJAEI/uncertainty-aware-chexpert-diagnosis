"""Tests for model architecture."""

import pytest
import torch

from uncertainty_aware_chexpert_diagnosis.models.model import (
    EvidentialLayer,
    EvidentialCheXpertModel,
    evidential_bce_loss,
)


def test_evidential_layer_initialization():
    """Test evidential layer initialization."""
    layer = EvidentialLayer(
        in_features=512,
        num_classes=5,
        evidence_activation='exp'
    )

    assert layer.num_classes == 5
    assert layer.evidence_activation == 'exp'


def test_evidential_layer_forward():
    """Test evidential layer forward pass."""
    batch_size = 8
    in_features = 512
    num_classes = 5

    layer = EvidentialLayer(in_features, num_classes)
    x = torch.randn(batch_size, in_features)

    outputs = layer(x)

    assert 'evidence' in outputs
    assert 'alpha' in outputs
    assert 'prob' in outputs
    assert 'epistemic' in outputs
    assert 'aleatoric' in outputs

    assert outputs['evidence'].shape == (batch_size, num_classes)
    assert outputs['alpha'].shape == (batch_size, num_classes)
    assert outputs['prob'].shape == (batch_size, num_classes)
    assert outputs['epistemic'].shape == (batch_size,)
    assert outputs['aleatoric'].shape == (batch_size,)

    # Check that probabilities sum to approximately 1
    prob_sums = outputs['prob'].sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5)


def test_evidential_layer_activations():
    """Test different evidence activations."""
    in_features = 512
    num_classes = 5
    x = torch.randn(4, in_features)

    for activation in ['exp', 'softplus', 'relu']:
        layer = EvidentialLayer(in_features, num_classes, activation)
        outputs = layer(x)

        # Evidence should be non-negative
        assert (outputs['evidence'] >= 0).all()


def test_model_initialization(config):
    """Test model initialization."""
    model = EvidentialCheXpertModel(config)

    assert model.num_classes == config['model']['num_classes']
    assert model.use_evidential == config['model']['use_evidential']


def test_model_forward_evidential(config, sample_batch):
    """Test model forward pass with evidential head."""
    config['model']['use_evidential'] = True
    model = EvidentialCheXpertModel(config)
    model.eval()

    with torch.no_grad():
        outputs = model(sample_batch['image'])

    batch_size = sample_batch['image'].shape[0]
    num_classes = config['model']['num_classes']

    assert 'prob' in outputs
    assert 'uncertainty_pred' in outputs
    assert 'epistemic' in outputs
    assert 'aleatoric' in outputs
    assert 'evidence' in outputs
    assert 'alpha' in outputs

    assert outputs['prob'].shape == (batch_size, num_classes)
    assert outputs['uncertainty_pred'].shape == (batch_size, num_classes)
    assert outputs['epistemic'].shape == (batch_size,)


def test_model_forward_standard(config, sample_batch):
    """Test model forward pass with standard head."""
    config['model']['use_evidential'] = False
    model = EvidentialCheXpertModel(config)
    model.eval()

    with torch.no_grad():
        outputs = model(sample_batch['image'])

    batch_size = sample_batch['image'].shape[0]
    num_classes = config['model']['num_classes']

    assert 'prob' in outputs
    assert 'uncertainty_pred' in outputs
    assert 'logits' in outputs

    assert outputs['prob'].shape == (batch_size, num_classes)
    assert outputs['logits'].shape == (batch_size, num_classes)


def test_model_parameter_count(config):
    """Test parameter counting."""
    model = EvidentialCheXpertModel(config)

    total_params = model.get_num_params()
    trainable_params = model.get_num_trainable_params()

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


def test_evidential_bce_loss():
    """Test evidential BCE loss."""
    batch_size = 8
    num_classes = 5

    alpha = torch.rand(batch_size, num_classes) * 10 + 1  # Alpha > 0
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    loss, loss_components = evidential_bce_loss(
        alpha,
        targets,
        epoch=5,
        kl_weight=0.01,
        kl_anneal_epochs=10
    )

    assert loss.item() >= 0
    assert 'bce_loss' in loss_components
    assert 'kl_div' in loss_components
    assert 'kl_weight' in loss_components

    # Check loss components are non-negative
    assert loss_components['bce_loss'].item() >= 0

    # Check KL annealing
    assert 0 <= loss_components['kl_weight'].item() <= 0.01


def test_model_gradient_flow(config, sample_batch):
    """Test that gradients flow through the model."""
    model = EvidentialCheXpertModel(config)
    model.train()

    # Enable gradient computation
    sample_batch['image'].requires_grad = True

    outputs = model(sample_batch['image'])

    # Compute simple loss
    loss = outputs['prob'].sum()
    loss.backward()

    # Check that some parameters have gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients found in model parameters"


def test_model_different_architectures(config):
    """Test model with different backbone architectures."""
    architectures = ['densenet121', 'resnet50', 'efficientnet_b0']

    for arch in architectures:
        config['model']['architecture'] = arch
        try:
            model = EvidentialCheXpertModel(config)
            assert model is not None
        except Exception as e:
            pytest.skip(f"Architecture {arch} not available: {e}")
