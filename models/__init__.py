"""
Models package for Temporal MAE.
"""

from .temporal_mae import TemporalMAE
from .pooling import build_pooling, GlobalAveragePooling, AttentionPooling, HybridPooling
from .losses import CombinedLoss, MAELoss, InfoNCELoss, ForecastingLoss, compute_anomaly_score
from .projection_heads import MAEReconstructionHead, ContrastiveProjectionHead, ForecastingHead

__all__ = [
    'TemporalMAE',
    'build_pooling',
    'GlobalAveragePooling',
    'AttentionPooling',
    'HybridPooling',
    'CombinedLoss',
    'MAELoss',
    'InfoNCELoss',
    'ForecastingLoss',
    'compute_anomaly_score',
    'MAEReconstructionHead',
    'ContrastiveProjectionHead',
    'ForecastingHead',
]
