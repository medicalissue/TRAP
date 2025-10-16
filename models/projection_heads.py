"""
Projection heads for MAE reconstruction, contrastive learning, and forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEReconstructionHead(nn.Module):
    """MLP head for MAE reconstruction."""

    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2

        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] encoded features
        Returns:
            [B, T, d_model] reconstructed features
        """
        return self.decoder(x)


class ContrastiveProjectionHead(nn.Module):
    """Projection head for contrastive learning (InfoNCE)."""

    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model] or [B, T, d_model] features
        Returns:
            [B, projection_dim] or [B, T, projection_dim] projected features
        """
        # Normalize to unit hypersphere
        proj = self.projection(x)
        return F.normalize(proj, dim=-1)


class ForecastingHead(nn.Module):
    """MLP head for forecasting future frames."""

    def __init__(self, d_model: int, num_future_steps: int = 4, hidden_dim: int = None):
        super().__init__()
        self.num_future_steps = num_future_steps

        if hidden_dim is None:
            hidden_dim = d_model * 2

        self.forecaster = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model * num_future_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model] last encoded feature
        Returns:
            [B, num_future_steps, d_model] predicted future features
        """
        B = x.size(0)
        out = self.forecaster(x)  # [B, d_model * num_future_steps]
        out = out.view(B, self.num_future_steps, -1)  # [B, num_future_steps, d_model]
        return out


class MultiHeadModel(nn.Module):
    """
    Combines multiple heads (MAE, contrastive, forecasting) on top of temporal encoder.
    """

    def __init__(
        self,
        d_model: int,
        use_mae: bool = True,
        use_contrastive: bool = True,
        use_forecasting: bool = False,
        projection_dim: int = 128,
        num_future_steps: int = 4,
    ):
        super().__init__()
        self.use_mae = use_mae
        self.use_contrastive = use_contrastive
        self.use_forecasting = use_forecasting

        # MAE reconstruction head
        if use_mae:
            self.mae_head = MAEReconstructionHead(d_model)

        # Contrastive projection head
        if use_contrastive:
            self.contrastive_head = ContrastiveProjectionHead(d_model, projection_dim)

        # Forecasting head
        if use_forecasting:
            self.forecasting_head = ForecastingHead(d_model, num_future_steps)

    def forward(self, encoded: torch.Tensor, return_contrastive_features: bool = False):
        """
        Args:
            encoded: [B, T, d_model] encoded sequence
            return_contrastive_features: whether to return per-token contrastive features

        Returns:
            Dictionary with head outputs
        """
        outputs = {}

        # MAE reconstruction
        if self.use_mae:
            outputs['reconstruction'] = self.mae_head(encoded)  # [B, T, d_model]

        # Contrastive learning
        if self.use_contrastive:
            if return_contrastive_features:
                # Project all tokens for temporal contrastive loss
                outputs['contrastive_features'] = self.contrastive_head(encoded)  # [B, T, proj_dim]
            else:
                # Project mean-pooled sequence for global contrastive loss
                pooled = encoded.mean(dim=1)  # [B, d_model]
                outputs['contrastive_features'] = self.contrastive_head(pooled)  # [B, proj_dim]

        # Forecasting
        if self.use_forecasting:
            # Use the last token to predict future
            last_token = encoded[:, -1, :]  # [B, d_model]
            outputs['forecast'] = self.forecasting_head(last_token)  # [B, num_steps, d_model]

        return outputs


if __name__ == "__main__":
    # Test projection heads
    batch_size = 8
    seq_len = 16
    d_model = 256
    projection_dim = 128
    num_future_steps = 4

    encoded = torch.randn(batch_size, seq_len, d_model)

    print("Testing projection heads...")

    # Test MAE head
    mae_head = MAEReconstructionHead(d_model)
    recon = mae_head(encoded)
    print(f"MAE reconstruction shape: {recon.shape}")
    assert recon.shape == (batch_size, seq_len, d_model)

    # Test contrastive head
    contrast_head = ContrastiveProjectionHead(d_model, projection_dim)
    contrast_out = contrast_head(encoded)
    print(f"Contrastive projection shape: {contrast_out.shape}")
    assert contrast_out.shape == (batch_size, seq_len, projection_dim)

    # Test forecasting head
    forecast_head = ForecastingHead(d_model, num_future_steps)
    last_token = encoded[:, -1, :]
    forecast_out = forecast_head(last_token)
    print(f"Forecasting shape: {forecast_out.shape}")
    assert forecast_out.shape == (batch_size, num_future_steps, d_model)

    # Test multi-head model
    multi_head = MultiHeadModel(
        d_model=d_model,
        use_mae=True,
        use_contrastive=True,
        use_forecasting=True,
        projection_dim=projection_dim,
        num_future_steps=num_future_steps,
    )

    outputs = multi_head(encoded, return_contrastive_features=True)
    print(f"Multi-head outputs: {list(outputs.keys())}")
    assert 'reconstruction' in outputs
    assert 'contrastive_features' in outputs
    assert 'forecast' in outputs

    print("All projection heads passed!")
