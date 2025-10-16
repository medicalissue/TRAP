"""
Feature pooling modules for object-level representations.
Supports GAP, Attention, and Hybrid pooling modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling over spatial dimensions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial features
        Returns:
            [B, d_model] pooled features
        """
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, C]

        # Project to d_model if needed
        if pooled.size(1) != self.d_model:
            if not hasattr(self, 'proj'):
                self.proj = nn.Linear(pooled.size(1), self.d_model).to(x.device)
            pooled = self.proj(pooled)

        return pooled


class AttentionPooling(nn.Module):
    """Attention-based pooling with learnable query."""

    def __init__(self, in_channels: int, d_model: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Learnable query token
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Input projection
        self.input_proj = nn.Linear(in_channels, d_model)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial features
        Returns:
            [B, d_model] pooled features
        """
        B, C, H, W = x.shape

        # Reshape to sequence: [B, H*W, C]
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Project to d_model
        x = self.input_proj(x)  # [B, H*W, d_model]

        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # [B, 1, d_model]

        # Apply attention: query attends to all spatial positions
        attn_out, attn_weights = self.attn(
            query, x, x,
            need_weights=True
        )  # attn_out: [B, 1, d_model]

        # Squeeze and normalize
        out = attn_out.squeeze(1)  # [B, d_model]
        out = self.norm(out)

        return out


class HybridPooling(nn.Module):
    """Hybrid pooling: concatenate GAP + Attention and project."""

    def __init__(self, in_channels: int, d_model: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # GAP branch
        self.gap = GlobalAveragePooling(d_model // 2)

        # Attention branch
        self.attn = AttentionPooling(in_channels, d_model // 2, num_heads, dropout)

        # Final projection to d_model
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial features
        Returns:
            [B, d_model] pooled features
        """
        # GAP branch
        gap_out = self.gap(x)  # [B, d_model//2]

        # Attention branch
        attn_out = self.attn(x)  # [B, d_model//2]

        # Concatenate
        combined = torch.cat([gap_out, attn_out], dim=-1)  # [B, d_model]

        # Final projection and normalization
        out = self.proj(combined)
        out = self.norm(out)

        return out


def build_pooling(mode: str, in_channels: int, d_model: int,
                  num_heads: int = 1, dropout: float = 0.1) -> nn.Module:
    """
    Factory function to build pooling module.

    Args:
        mode: 'gap' | 'attn' | 'hybrid'
        in_channels: Number of input channels
        d_model: Output dimension
        num_heads: Number of attention heads (for attn/hybrid)
        dropout: Dropout rate (for attn/hybrid)

    Returns:
        Pooling module
    """
    if mode == 'gap':
        return GlobalAveragePooling(d_model)
    elif mode == 'attn':
        return AttentionPooling(in_channels, d_model, num_heads, dropout)
    elif mode == 'hybrid':
        return HybridPooling(in_channels, d_model, num_heads, dropout)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}. Choose from ['gap', 'attn', 'hybrid']")


if __name__ == "__main__":
    # Test pooling modules
    batch_size = 4
    in_channels = 512
    height, width = 7, 7
    d_model = 256

    x = torch.randn(batch_size, in_channels, height, width)

    print("Testing pooling modules...")
    print(f"Input shape: {x.shape}")

    # Test GAP
    gap = build_pooling('gap', in_channels, d_model)
    gap_out = gap(x)
    print(f"GAP output shape: {gap_out.shape}")
    assert gap_out.shape == (batch_size, d_model)

    # Test Attention
    attn = build_pooling('attn', in_channels, d_model)
    attn_out = attn(x)
    print(f"Attention output shape: {attn_out.shape}")
    assert attn_out.shape == (batch_size, d_model)

    # Test Hybrid
    hybrid = build_pooling('hybrid', in_channels, d_model)
    hybrid_out = hybrid(x)
    print(f"Hybrid output shape: {hybrid_out.shape}")
    assert hybrid_out.shape == (batch_size, d_model)

    print("All pooling modules passed!")
