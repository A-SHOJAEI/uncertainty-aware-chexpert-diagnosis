"""Training loop for evidential CheXpert model."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from uncertainty_aware_chexpert_diagnosis.models.model import evidential_bce_loss
from uncertainty_aware_chexpert_diagnosis.evaluation.metrics import UncertaintyMetrics

logger = logging.getLogger(__name__)


class EvidentialTrainer:
    """Trainer for uncertainty-aware CheXpert model."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.training_config = config.get('training', {})
        self.val_config = config.get('validation', {})
        self.model_config = config.get('model', {})

        # Training parameters
        self.epochs = self.training_config.get('epochs', 50)
        self.learning_rate = self.training_config.get('learning_rate', 0.0001)
        self.weight_decay = self.training_config.get('weight_decay', 0.00001)
        self.gradient_clip = self.training_config.get('gradient_clip', 1.0)
        self.use_amp = self.training_config.get('use_amp', True)

        # Loss parameters
        self.kl_weight = self.training_config.get('kl_weight', 0.01)
        self.kl_anneal_epochs = self.training_config.get('kl_anneal_epochs', 10)
        self.epistemic_weight = self.model_config.get('epistemic_weight', 1.0)
        self.aleatoric_weight = self.model_config.get('aleatoric_weight', 1.0)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Metrics
        self.metrics = UncertaintyMetrics(config)

        # Checkpointing
        self.checkpoint_dir = Path(config.get('experiment', {}).get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = self.val_config.get('early_stopping_patience', 10)

        logger.info(f"Initialized trainer with learning_rate={self.learning_rate}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_type = self.training_config.get('optimizer', 'adam').lower()

        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = self.training_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.training_config.get('scheduler', 'cosine').lower()

        if scheduler_type == 'cosine':
            min_lr = self.training_config.get('min_lr', 0.000001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            step_size = self.training_config.get('step_size', 10)
            gamma = self.training_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        return scheduler

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_unc_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            uncertainty_mask = batch['uncertainty_mask'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss, loss_dict = self._compute_loss(
                        outputs, labels, uncertainty_mask, epoch
                    )
            else:
                outputs = self.model(images)
                loss, loss_dict = self._compute_loss(
                    outputs, labels, uncertainty_mask, epoch
                )

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_cls_loss += loss_dict['cls_loss'].item()
            total_unc_loss += loss_dict['unc_loss'].item()
            if 'kl_div' in loss_dict:
                total_kl_loss += loss_dict['kl_div'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{loss_dict['cls_loss'].item():.4f}",
                'unc': f"{loss_dict['unc_loss'].item():.4f}"
            })

        # Average metrics
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_cls_loss': total_cls_loss / num_batches,
            'train_unc_loss': total_unc_loss / num_batches,
            'train_kl_loss': total_kl_loss / num_batches if total_kl_loss > 0 else 0.0
        }

        return metrics

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        uncertainty_mask: torch.Tensor,
        epoch: int
    ) -> tuple:
        """
        Compute training loss.

        Args:
            outputs: Model outputs
            labels: Ground truth labels
            uncertainty_mask: Binary mask for uncertain labels
            epoch: Current epoch

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Classification loss
        if 'alpha' in outputs:
            # Evidential loss
            cls_loss, loss_components = evidential_bce_loss(
                outputs['alpha'],
                labels,
                epoch,
                self.kl_weight,
                self.kl_anneal_epochs
            )
        else:
            # Standard BCE loss
            # Disable autocast for BCE with probabilities (not logits)
            with torch.amp.autocast('cuda', enabled=False):
                prob = outputs['prob'].float()
                labels_float = labels.float()
                cls_loss = nn.functional.binary_cross_entropy(
                    prob,
                    labels_float,
                    reduction='mean'
                )
            loss_components = {}

        # Uncertainty prediction loss (novel: learn to predict radiologist uncertainty)
        unc_pred = outputs['uncertainty_pred']
        # Disable autocast for BCE with probabilities (not logits)
        with torch.amp.autocast('cuda', enabled=False):
            unc_pred_float = unc_pred.float()
            uncertainty_mask_float = uncertainty_mask.float()
            unc_loss = nn.functional.binary_cross_entropy(
                unc_pred_float,
                uncertainty_mask_float,
                reduction='mean'
            )

        # Combined loss
        total_loss = cls_loss + unc_loss

        loss_dict = {
            'cls_loss': cls_loss,
            'unc_loss': unc_loss,
            **loss_components
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        all_probs = []
        all_labels = []
        all_uncertainty_masks = []
        all_epistemic = []
        all_aleatoric = []

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            uncertainty_mask = batch['uncertainty_mask'].to(self.device)

            # Forward pass
            outputs = self.model(images)

            # Compute loss
            loss, _ = self._compute_loss(outputs, labels, uncertainty_mask, epoch)
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            all_probs.append(outputs['prob'].cpu())
            all_labels.append(labels.cpu())
            all_uncertainty_masks.append(uncertainty_mask.cpu())

            if 'epistemic' in outputs:
                all_epistemic.append(outputs['epistemic'].cpu())
            if 'aleatoric' in outputs:
                all_aleatoric.append(outputs['aleatoric'].cpu())

        # Concatenate all batches
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_uncertainty_masks = torch.cat(all_uncertainty_masks, dim=0)

        if len(all_epistemic) > 0:
            all_epistemic = torch.cat(all_epistemic, dim=0)
        else:
            all_epistemic = None

        if len(all_aleatoric) > 0:
            all_aleatoric = torch.cat(all_aleatoric, dim=0)
        else:
            all_aleatoric = None

        # Compute metrics
        metrics = self.metrics.compute_metrics(
            all_probs.numpy(),
            all_labels.numpy(),
            all_uncertainty_masks.numpy(),
            all_epistemic.numpy() if all_epistemic is not None else None,
            all_aleatoric.numpy() if all_aleatoric is not None else None
        )

        metrics['val_loss'] = total_loss / num_batches

        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting training")

        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Log training metrics
            logger.info(f"Epoch {epoch} - Train loss: {train_metrics['train_loss']:.4f}")

            # Validate
            if epoch % self.val_config.get('eval_interval', 1) == 0:
                val_metrics = self.validate(val_loader, epoch)

                # Log validation metrics
                logger.info(f"Epoch {epoch} - Val loss: {val_metrics['val_loss']:.4f}")
                logger.info(f"Epoch {epoch} - Val AUROC: {val_metrics.get('auroc_mean', 0):.4f}")

                # Check for best model
                current_metric = val_metrics.get(
                    self.val_config.get('metric_for_best', 'auroc_mean'),
                    0.0
                )

                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0

                    if self.val_config.get('save_best', True):
                        self.save_checkpoint(epoch, val_metrics, is_best=True)
                        logger.info(f"Saved best model with {self.val_config.get('metric_for_best', 'auroc_mean')}={current_metric:.4f}")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        logger.info("Training complete")

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
