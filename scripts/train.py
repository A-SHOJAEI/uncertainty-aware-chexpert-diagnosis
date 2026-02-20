#!/usr/bin/env python
"""Training script for uncertainty-aware CheXpert diagnosis."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import random

import numpy as np
import torch

from uncertainty_aware_chexpert_diagnosis.data.loader import CheXpertDataModule
from uncertainty_aware_chexpert_diagnosis.models.model import EvidentialCheXpertModel
from uncertainty_aware_chexpert_diagnosis.training.trainer import EvidentialTrainer
from uncertainty_aware_chexpert_diagnosis.utils.config import (
    load_config,
    setup_logging,
    set_seed,
    create_directories,
    get_device,
    count_parameters,
    save_config,
)

logger = logging.getLogger(__name__)


def setup_mlflow(config: dict) -> None:
    """
    Setup MLflow tracking with error handling.

    Args:
        config: Configuration dictionary
    """
    try:
        import mlflow

        mlflow_config = config.get('mlflow', {})
        tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
        experiment_name = mlflow_config.get('experiment_name', 'chexpert_uncertainty')

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking enabled: {tracking_uri}")
        return True
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")
        logger.warning("Continuing without MLflow tracking")
        return False


def log_to_mlflow(metrics: dict, step: int = None) -> None:
    """
    Log metrics to MLflow with error handling.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
    """
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass


def main() -> None:
    """Main training function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    config = load_config(str(config_path))

    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting uncertainty-aware CheXpert training")

    # Set random seeds for reproducibility
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Create output directories
    create_directories(config)

    # Setup MLflow
    mlflow_enabled = setup_mlflow(config)

    # Get device
    device = get_device()

    try:
        if mlflow_enabled:
            import mlflow
            with mlflow.start_run():
                # Log config
                try:
                    mlflow.log_params({
                        'seed': seed,
                        'architecture': config.get('model', {}).get('architecture', 'densenet121'),
                        'learning_rate': config.get('training', {}).get('learning_rate', 0.0001),
                        'batch_size': config.get('training', {}).get('batch_size', 32),
                        'epochs': config.get('training', {}).get('epochs', 50),
                    })
                except Exception as e:
                    logger.warning(f"Could not log params to MLflow: {e}")

                # Run training
                train_model(config, device, mlflow_enabled)
        else:
            # Run without MLflow
            train_model(config, device, mlflow_enabled=False)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


def train_model(config: dict, device: torch.device, mlflow_enabled: bool = False) -> None:
    """
    Train the model.

    Args:
        config: Configuration dictionary
        device: Device to train on
        mlflow_enabled: Whether MLflow is available
    """
    # Initialize data module
    logger.info("Initializing data module")
    data_module = CheXpertDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    # Create dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    logger.info(f"Training samples: {len(data_module.train_dataset)}")
    logger.info(f"Validation samples: {len(data_module.val_dataset)}")

    # Initialize model
    logger.info("Initializing model")
    model = EvidentialCheXpertModel(config)
    model = model.to(device)

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    if mlflow_enabled:
        try:
            import mlflow
            mlflow.log_param('num_parameters', num_params)
        except Exception:
            pass

    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = EvidentialTrainer(model, config, device)

    # Train
    logger.info("Starting training loop")
    epochs = config.get('training', {}).get('epochs', 50)

    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*50}")

        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Log training metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Train Classification Loss: {train_metrics['train_cls_loss']:.4f}")
        logger.info(f"Train Uncertainty Loss: {train_metrics['train_unc_loss']:.4f}")

        if mlflow_enabled:
            log_to_mlflow(train_metrics, step=epoch)

        # Validate
        val_config = config.get('validation', {})
        if epoch % val_config.get('eval_interval', 1) == 0:
            val_metrics = trainer.validate(val_loader, epoch)

            # Log validation metrics
            logger.info(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Validation AUROC (mean): {val_metrics.get('auroc_mean', 0.0):.4f}")
            logger.info(f"AUROC (certain labels): {val_metrics.get('auroc_certain_mean', 0.0):.4f}")
            logger.info(f"AUROC (uncertain labels): {val_metrics.get('auroc_uncertain_mean', 0.0):.4f}")
            logger.info(f"Calibration ECE: {val_metrics.get('calibration_ece', 0.0):.4f}")
            logger.info(f"Uncertainty Correlation: {val_metrics.get('uncertainty_correlation', 0.0):.4f}")

            if mlflow_enabled:
                log_to_mlflow(val_metrics, step=epoch)

            # Check for best model
            metric_for_best = val_config.get('metric_for_best', 'auroc_mean')
            current_metric = val_metrics.get(metric_for_best, 0.0)

            if current_metric > trainer.best_metric:
                trainer.best_metric = current_metric
                trainer.patience_counter = 0

                if val_config.get('save_best', True):
                    trainer.save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"✓ Saved best model with {metric_for_best}={current_metric:.4f}")
            else:
                trainer.patience_counter += 1

            # Early stopping check
            if trainer.patience_counter >= trainer.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Step scheduler
        if trainer.scheduler is not None:
            trainer.scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")

    # Save final model
    checkpoint_dir = Path(config.get('experiment', {}).get('checkpoint_dir', 'checkpoints'))
    final_path = checkpoint_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    # Log best metrics
    logger.info(f"\n{'='*50}")
    logger.info("Training Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Best {val_config.get('metric_for_best', 'auroc_mean')}: {trainer.best_metric:.4f}")
    logger.info(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
