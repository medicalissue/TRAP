"""
Loss functions for Temporal MAE training.
Includes MAE reconstruction, InfoNCE contrastive, and forecasting losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MAELoss(nn.Module):
    """
    Masked Autoencoder reconstruction loss.
    Compute MSE only on masked positions.
    """

    def __init__(self, norm_pix_loss: bool = False):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prediction: [B, T, d_model] reconstructed features
            target: [B, T, d_model] original features
            mask: [B, T] boolean mask (True for masked positions)

        Returns:
            Scalar loss
        """
        if self.norm_pix_loss:
            # Normalize target features per sample
            mean = target.mean(dim=-1, keepdim=True)
            std = target.std(dim=-1, keepdim=True)
            target = (target - mean) / (std + 1e-6)

        # Compute MSE
        loss = (prediction - target) ** 2
        loss = loss.mean(dim=-1)  # [B, T] mean over feature dim

        # Apply mask: only compute loss on masked positions
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for temporal consistency.
    Encourages adjacent frames to have similar representations.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, T, d_proj] normalized projected features
            valid_mask: [B, T] boolean mask for valid positions

        Returns:
            Scalar loss
        """
        B, T, D = features.shape

        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=features.device)

        total_loss = 0.0
        total_pairs = 0

        # Compute contrastive loss between adjacent frames
        for t in range(T - 1):
            # Get current and next frame features
            curr = features[:, t, :]  # [B, d_proj]
            next_frame = features[:, t + 1, :]  # [B, d_proj]

            # Check which samples have valid pairs
            valid_pairs = valid_mask[:, t] & valid_mask[:, t + 1]  # [B]

            if valid_pairs.sum() == 0:
                continue

            # Select only valid pairs
            curr = curr[valid_pairs]  # [N, d_proj]
            next_frame = next_frame[valid_pairs]  # [N, d_proj]
            N = curr.size(0)

            # Compute similarity matrix
            # Positive pairs: (curr[i], next[i])
            # Negative pairs: (curr[i], next[j]) for j != i
            pos_sim = (curr * next_frame).sum(dim=-1) / self.temperature  # [N]

            # Compute all pairwise similarities
            all_sim = torch.matmul(curr, next_frame.T) / self.temperature  # [N, N]

            # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
            loss = -pos_sim + torch.logsumexp(all_sim, dim=1)
            total_loss += loss.sum()
            total_pairs += N

        if total_pairs == 0:
            return torch.tensor(0.0, device=features.device)

        return total_loss / total_pairs


class ForecastingLoss(nn.Module):
    """
    Forecasting loss for predicting future frames.
    MSE between predicted and actual future features.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prediction: [B, num_steps, d_model] predicted future features
            target: [B, num_steps, d_model] actual future features
            valid_mask: [B, num_steps] boolean mask for valid positions

        Returns:
            Scalar loss
        """
        if valid_mask is None:
            valid_mask = torch.ones(
                prediction.size(0), prediction.size(1),
                dtype=torch.bool, device=prediction.device
            )

        # Compute MSE
        loss = (prediction - target) ** 2
        loss = loss.mean(dim=-1)  # [B, num_steps] mean over feature dim

        # Apply mask
        loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for Temporal MAE training.
    L_total = λ_mae * L_mae + λ_cl * L_cl + λ_forecast * L_forecast
    """

    def __init__(
        self,
        lambda_mae: float = 1.0,
        lambda_cl: float = 0.3,
        lambda_forecast: float = 0.1,
        temperature: float = 0.07,
        norm_pix_loss: bool = False,
    ):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_cl = lambda_cl
        self.lambda_forecast = lambda_forecast

        self.mae_loss = MAELoss(norm_pix_loss=norm_pix_loss)
        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        self.forecasting_loss = ForecastingLoss()

    def forward(
        self,
        outputs: dict,
        target: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss from model outputs.

        Args:
            outputs: Dictionary from TemporalMAE forward pass
            target: [B, T, d_model] original input features
            future_target: [B, num_steps, d_model] future features (if forecasting)

        Returns:
            Dictionary with all losses
        """
        losses = {}
        total_loss = 0.0

        # MAE reconstruction loss
        if 'reconstruction' in outputs and outputs['mask'] is not None:
            mae_loss = self.mae_loss(
                outputs['reconstruction'],
                target,
                outputs['mask']
            )
            losses['mae_loss'] = mae_loss
            total_loss += self.lambda_mae * mae_loss

        # Contrastive loss
        if 'contrastive_features' in outputs:
            cl_loss = self.contrastive_loss(
                outputs['contrastive_features'],
                outputs['valid_mask']
            )
            losses['contrastive_loss'] = cl_loss
            total_loss += self.lambda_cl * cl_loss

        # Forecasting loss
        if 'forecast' in outputs and future_target is not None:
            forecast_loss = self.forecasting_loss(
                outputs['forecast'],
                future_target,
                valid_mask=None  # Assume future positions are valid
            )
            losses['forecasting_loss'] = forecast_loss
            total_loss += self.lambda_forecast * forecast_loss

        losses['total_loss'] = total_loss

        return losses


def compute_anomaly_score(
    reconstruction_error: torch.Tensor,
    temporal_similarity: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute anomaly score from reconstruction error and temporal similarity.

    Args:
        reconstruction_error: [B] or [B, T] reconstruction error
        temporal_similarity: [B] or [B, T] temporal similarity (0-1)
        alpha: balance factor (0-1)

    Returns:
        [B] or [B, T] anomaly scores
    """
    # Normalize reconstruction error to [0, 1] range
    recon_norm = torch.sigmoid(reconstruction_error)

    # Anomaly score: high when reconstruction is poor OR similarity is low
    anomaly_score = alpha * recon_norm + (1 - alpha) * (1 - temporal_similarity)

    return anomaly_score


if __name__ == "__main__":
    # Test loss functions
    batch_size = 8
    seq_len = 16
    d_model = 256
    projection_dim = 128

    print("Testing loss functions...")

    # Create dummy data
    prediction = torch.randn(batch_size, seq_len, d_model)
    target = torch.randn(batch_size, seq_len, d_model)
    mask = torch.rand(batch_size, seq_len) > 0.75  # Random mask
    valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Test MAE loss
    mae_loss_fn = MAELoss()
    mae_loss = mae_loss_fn(prediction, target, mask)
    print(f"MAE loss: {mae_loss.item():.4f}")
    assert mae_loss.numel() == 1

    # Test InfoNCE loss
    contrastive_features = F.normalize(torch.randn(batch_size, seq_len, projection_dim), dim=-1)
    infonce_loss_fn = InfoNCELoss(temperature=0.07)
    infonce_loss = infonce_loss_fn(contrastive_features, valid_mask)
    print(f"InfoNCE loss: {infonce_loss.item():.4f}")
    assert infonce_loss.numel() == 1

    # Test forecasting loss
    forecast_pred = torch.randn(batch_size, 4, d_model)
    forecast_target = torch.randn(batch_size, 4, d_model)
    forecast_loss_fn = ForecastingLoss()
    forecast_loss = forecast_loss_fn(forecast_pred, forecast_target)
    print(f"Forecasting loss: {forecast_loss.item():.4f}")
    assert forecast_loss.numel() == 1

    # Test combined loss
    outputs = {
        'reconstruction': prediction,
        'contrastive_features': contrastive_features,
        'forecast': forecast_pred,
        'mask': mask,
        'valid_mask': valid_mask,
    }

    combined_loss_fn = CombinedLoss(lambda_mae=1.0, lambda_cl=0.3, lambda_forecast=0.1)
    losses = combined_loss_fn(outputs, target, forecast_target)
    print(f"\nCombined losses:")
    for key, val in losses.items():
        print(f"  {key}: {val.item():.4f}")

    # Test anomaly score computation
    recon_error = torch.randn(batch_size).abs()
    temp_similarity = torch.rand(batch_size)
    anomaly_scores = compute_anomaly_score(recon_error, temp_similarity, alpha=0.5)
    print(f"\nAnomaly scores shape: {anomaly_scores.shape}")
    print(f"Anomaly scores range: [{anomaly_scores.min().item():.4f}, {anomaly_scores.max().item():.4f}]")

    print("\nAll loss functions passed!")
