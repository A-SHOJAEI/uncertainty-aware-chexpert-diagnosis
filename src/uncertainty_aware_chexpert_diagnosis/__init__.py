"""
Uncertainty-Aware CheXpert Diagnosis

Multi-label chest X-ray diagnosis with evidential deep learning for uncertainty quantification.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from uncertainty_aware_chexpert_diagnosis.data.loader import CheXpertDataModule
from uncertainty_aware_chexpert_diagnosis.models.model import EvidentialCheXpertModel
from uncertainty_aware_chexpert_diagnosis.training.trainer import EvidentialTrainer
from uncertainty_aware_chexpert_diagnosis.evaluation.metrics import UncertaintyMetrics

__all__ = [
    "CheXpertDataModule",
    "EvidentialCheXpertModel",
    "EvidentialTrainer",
    "UncertaintyMetrics",
]
