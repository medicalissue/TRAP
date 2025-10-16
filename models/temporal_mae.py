"""
Temporal Masked Autoencoder (MAE) with Contrastive Learning.
Encoder-only transformer that processes object-level temporal sequences.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .projection_heads import MultiHeadModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with pre-norm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            mask: [T, T] attention mask
            key_padding_mask: [B, T] padding mask (True for positions to ignore)
        Returns:
            [B, T, d_model]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class TemporalMAE(nn.Module):
    """
    Temporal Masked Autoencoder with Contrastive Learning.
    Processes sequences of object-level features.
    """

    def __init__(
        self,
        d_model: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        mask_ratio: float = 0.75,
        max_seq_len: int = 512,
        use_mae: bool = True,
        use_contrastive: bool = True,
        use_forecasting: bool = False,
        projection_dim: int = 128,
        num_future_steps: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.use_mae = use_mae
        self.use_contrastive = use_contrastive
        self.use_forecasting = use_forecasting

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Multi-head model (MAE, contrastive, forecasting)
        self.heads = MultiHeadModel(
            d_model=d_model,
            use_mae=use_mae,
            use_contrastive=use_contrastive,
            use_forecasting=use_forecasting,
            projection_dim=projection_dim,
            num_future_steps=num_future_steps,
        )

        # Mask token for MAE
        if use_mae:
            self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking for MAE.
        Only mask valid (non-padding) positions.

        Args:
            x: [B, T, d_model]
            mask_ratio: percentage of tokens to mask
            valid_mask: [B, T] boolean mask (True for valid positions)

        Returns:
            x_masked: [B, T, d_model] with masked positions replaced by mask_token
            mask: [B, T] boolean mask (True for masked positions)
            ids_restore: [B, T] indices to restore original order
        """
        B, T, D = x.shape

        # If no valid_mask provided, assume all positions are valid
        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        # Count valid tokens per sample
        num_valid = valid_mask.sum(dim=1)  # [B]

        # Generate random noise for shuffling (only for valid positions)
        noise = torch.rand(B, T, device=x.device)
        noise[~valid_mask] = 2.0  # Set invalid positions to high value so they're not selected

        # Sort noise to get shuffling indices
        ids_shuffle = torch.argsort(noise, dim=1)  # [B, T]
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # [B, T]

        # Calculate number of tokens to keep
        num_keep = (num_valid * (1 - mask_ratio)).long()  # [B]

        # Create mask: True for masked positions
        mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        for i in range(B):
            mask[i, :num_keep[i]] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)  # Unshuffle

        # Don't mask invalid positions
        mask = mask & valid_mask

        # Apply masking
        x_masked = x.clone()
        mask_token = self.mask_token.expand(B, T, -1)
        x_masked[mask] = mask_token[mask]

        return x_masked, mask, ids_restore

    def forward(
        self,
        x: torch.Tensor,
        apply_masking: bool = True,
        valid_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ):
        """
        Forward pass through Temporal MAE.

        Args:
            x: [B, T, d_model] input sequence
            apply_masking: whether to apply random masking (training mode)
            valid_mask: [B, T] boolean mask for valid positions
            return_all_tokens: return all tokens or just the mean-pooled output

        Returns:
            Dictionary with model outputs
        """
        B, T, D = x.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # Apply masking if in training mode
        mask = None
        if apply_masking and self.use_mae:
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio, valid_mask)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create key padding mask for transformer (True for positions to ignore)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask  # Invert: True for invalid positions

        # Pass through transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        # Final normalization
        x = self.norm(x)

        # Pass through heads
        outputs = self.heads(x, return_contrastive_features=True)

        # Add mask and encoded features to outputs
        outputs['encoded'] = x
        outputs['mask'] = mask
        outputs['valid_mask'] = valid_mask

        # Optionally return mean-pooled representation
        if not return_all_tokens:
            if valid_mask is not None:
                # Mean pool only over valid positions
                valid_sum = x.sum(dim=1)  # [B, d_model]
                valid_count = valid_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
                outputs['pooled'] = valid_sum / valid_count.clamp(min=1.0)
            else:
                outputs['pooled'] = x.mean(dim=1)  # [B, d_model]

        return outputs


if __name__ == "__main__":
    # Test Temporal MAE
    batch_size = 8
    seq_len = 16
    d_model = 256

    x = torch.randn(batch_size, seq_len, d_model)
    valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Simulate some missing positions
    valid_mask[0, 10:] = False
    valid_mask[1, 8:] = False

    print("Testing Temporal MAE...")
    print(f"Input shape: {x.shape}")
    print(f"Valid mask shape: {valid_mask.shape}")

    model = TemporalMAE(
        d_model=d_model,
        depth=6,
        num_heads=8,
        mask_ratio=0.75,
        use_mae=True,
        use_contrastive=True,
        use_forecasting=True,
    )

    # Training mode (with masking)
    outputs = model(x, apply_masking=True, valid_mask=valid_mask)
    print("\nTraining mode outputs:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {val}")

    # Inference mode (no masking)
    outputs = model(x, apply_masking=False, valid_mask=valid_mask)
    print("\nInference mode outputs:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")

    print("\nTemporal MAE passed!")
