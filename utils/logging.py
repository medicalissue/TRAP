"""
Logging utilities for wandb integration.
"""

import wandb
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import io
from PIL import Image


class WandbLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Initialize wandb logger.

        Args:
            config: Configuration dictionary
            enabled: Whether to enable logging
        """
        self.enabled = enabled and config.get('mode', 'online') != 'disabled'

        if self.enabled:
            wandb.init(
                project=config.get('project', 'temporal_mae_ood'),
                entity=config.get('entity', None),
                name=config.get('name', None),
                config=config,
                mode=config.get('mode', 'online'),
                tags=config.get('tags', []),
                notes=config.get('notes', ''),
            )
            print(f"Initialized wandb run: {wandb.run.name}")
        else:
            print("Wandb logging disabled")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_image(self, key: str, image: np.ndarray, step: Optional[int] = None):
        """Log image to wandb."""
        if self.enabled:
            wandb.log({key: wandb.Image(image)}, step=step)

    def log_video(self, key: str, video_path: str, step: Optional[int] = None):
        """Log video to wandb."""
        if self.enabled:
            wandb.log({key: wandb.Video(video_path)}, step=step)

    def log_table(self, key: str, data: list, columns: list, step: Optional[int] = None):
        """Log table to wandb."""
        if self.enabled:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({key: table}, step=step)

    def log_artifact(self, artifact_name: str, artifact_type: str, file_path: str):
        """Log artifact to wandb."""
        if self.enabled:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)

    def watch_model(self, model: torch.nn.Module, log: str = "gradients", log_freq: int = 100):
        """Watch model gradients and parameters."""
        if self.enabled:
            wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        """Finish wandb run."""
        if self.enabled:
            wandb.finish()


def plot_to_image(figure):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)


def plot_loss_curves(losses: Dict[str, list], save_path: Optional[str] = None):
    """
    Plot training loss curves.

    Args:
        losses: Dictionary of loss lists
        save_path: Optional path to save plot

    Returns:
        PIL Image of plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, values in losses.items():
        ax.plot(values, label=name)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=100)

    img = plot_to_image(fig)
    plt.close(fig)

    return img


def plot_attention_map(attention_weights: np.ndarray, save_path: Optional[str] = None):
    """
    Plot attention weights as heatmap.

    Args:
        attention_weights: [T, T] attention matrix
        save_path: Optional path to save plot

    Returns:
        PIL Image of plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Attention Weights')
    plt.colorbar(im, ax=ax)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=100)

    img = plot_to_image(fig)
    plt.close(fig)

    return img


def plot_reconstruction(original: np.ndarray, reconstructed: np.ndarray,
                       save_path: Optional[str] = None):
    """
    Plot original vs reconstructed features.

    Args:
        original: [T, d_model] original features
        reconstructed: [T, d_model] reconstructed features
        save_path: Optional path to save plot

    Returns:
        PIL Image of plot
    """
    T = original.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Original
    im1 = axes[0].imshow(original.T, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Feature Dim')
    axes[0].set_title('Original Features')
    plt.colorbar(im1, ax=axes[0])

    # Reconstructed
    im2 = axes[1].imshow(reconstructed.T, cmap='viridis', aspect='auto')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Feature Dim')
    axes[1].set_title('Reconstructed Features')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=100)

    img = plot_to_image(fig)
    plt.close(fig)

    return img


def create_summary_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Create a markdown summary table from results.

    Args:
        results: Nested dictionary of results {dataset: {metric: value}}

    Returns:
        Markdown table string
    """
    if not results:
        return "No results available."

    # Get all unique metrics
    all_metrics = set()
    for dataset_results in results.values():
        all_metrics.update(dataset_results.keys())
    metrics = sorted(all_metrics)

    # Create header
    table = "| Dataset | " + " | ".join(metrics) + " |\n"
    table += "|---------|" + "|".join(["--------"] * len(metrics)) + "|\n"

    # Add rows
    for dataset, dataset_results in sorted(results.items()):
        row = f"| {dataset} |"
        for metric in metrics:
            value = dataset_results.get(metric, "N/A")
            if isinstance(value, float):
                row += f" {value:.4f} |"
            else:
                row += f" {value} |"
        table += row + "\n"

    return table
