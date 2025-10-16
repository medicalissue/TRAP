"""
Training script for Temporal MAE with Contrastive Learning.
Trains on normal-only sequences from multiple datasets.
"""

import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from pathlib import Path
from typing import Dict
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np

from models.temporal_mae import TemporalMAE
from models.losses import CombinedLoss
from data.trackseq_dataset import create_dataloader
from utils.seed import set_seed, worker_init_fn
from utils.device import resolve_device
from utils.logging import WandbLogger, plot_loss_curves, plot_reconstruction


class Trainer:
    """Trainer for Temporal MAE."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg)

        # Set random seed
        set_seed(cfg.seed)

        # Initialize wandb
        self.logger = WandbLogger(
            config=OmegaConf.to_container(cfg, resolve=True),
            enabled=(cfg.wandb.mode != 'disabled')
        )

        # Build model
        self.model = self.build_model()
        self.model.to(self.device)

        # Build loss
        self.criterion = CombinedLoss(
            lambda_mae=cfg.train.losses.lambda_mae,
            lambda_cl=cfg.train.losses.lambda_cl,
            lambda_forecast=cfg.train.losses.lambda_forecast,
            temperature=cfg.model.contrastive.temperature,
        )

        # Build optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Build dataloaders
        self.train_loader = self.build_dataloader('train')
        if cfg.train.validation.enabled:
            self.val_loader = self.build_dataloader('val')
        else:
            self.val_loader = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Loss history
        self.train_losses = {
            'total': [],
            'mae': [],
            'contrastive': [],
            'forecasting': [],
        }
        self.val_losses = {
            'total': [],
        }

        # Checkpoint directory
        self.checkpoint_dir = Path(cfg.paths.checkpoints)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def build_model(self) -> nn.Module:
        """Build Temporal MAE model."""
        model = TemporalMAE(
            d_model=self.cfg.model.temporal.d_model,
            depth=self.cfg.model.temporal.depth,
            num_heads=self.cfg.model.temporal.num_heads,
            mlp_ratio=self.cfg.model.temporal.mlp_ratio,
            dropout=self.cfg.model.temporal.dropout,
            attn_dropout=self.cfg.model.temporal.attn_dropout,
            mask_ratio=self.cfg.model.temporal.mask_ratio,
            max_seq_len=self.cfg.model.temporal.seq_length * 2,
            use_mae=self.cfg.model.heads.use_mae,
            use_contrastive=self.cfg.model.heads.use_contrastive,
            use_forecasting=self.cfg.model.heads.use_forecasting,
            projection_dim=self.cfg.model.contrastive.projection_dim,
            num_future_steps=self.cfg.model.heads.forecasting_steps,
        )
        return model

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer."""
        if self.cfg.train.optimizer.name.lower() == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
                betas=tuple(self.cfg.train.optimizer.betas),
                eps=self.cfg.train.optimizer.eps,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.train.optimizer.name}")

        return optimizer

    def build_scheduler(self):
        """Build learning rate scheduler."""
        if self.cfg.train.scheduler.name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.train.epochs - self.cfg.train.scheduler.warmup_epochs,
                eta_min=self.cfg.train.scheduler.min_lr,
            )
        elif self.cfg.train.scheduler.name.lower() == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=self.cfg.train.scheduler.step_size,
                gamma=self.cfg.train.scheduler.gamma,
            )
        elif self.cfg.train.scheduler.name.lower() == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.train.scheduler.name}")

        return scheduler

    def build_dataloader(self, split: str):
        """Build dataloader for train or validation."""
        # For validation, use a subset of training data
        if split == 'val':
            split_name = 'train'  # Load from train but will be subset
        else:
            split_name = split

        dataloader = create_dataloader(
            feature_dir=Path(self.cfg.data.features_root),
            split=split_name,
            datasets=self.cfg.data.datasets,
            batch_size=self.cfg.train.batch_size,
            seq_length=self.cfg.data.sequence.T,
            num_workers=self.cfg.num_workers,
            shuffle=(split == 'train'),
            augment=(split == 'train' and self.cfg.train.use_augmentation),
        )

        return dataloader

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total': [],
            'mae': [],
            'contrastive': [],
            'forecasting': [],
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.cfg.train.epochs}")

        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)  # [B, T, d_model]
            valid_mask = batch['valid_mask'].to(self.device)  # [B, T]

            # Forward pass
            outputs = self.model(
                features,
                apply_masking=True,
                valid_mask=valid_mask,
            )

            # Compute losses
            losses = self.criterion(outputs, features)

            total_loss = losses['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Record losses
            epoch_losses['total'].append(total_loss.item())
            if 'mae_loss' in losses:
                epoch_losses['mae'].append(losses['mae_loss'].item())
            if 'contrastive_loss' in losses:
                epoch_losses['contrastive'].append(losses['contrastive_loss'].item())
            if 'forecasting_loss' in losses:
                epoch_losses['forecasting'].append(losses['forecasting_loss'].item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # Log to wandb
            if self.global_step % self.cfg.train.logging.log_every_n_steps == 0:
                log_dict = {
                    'train/total_loss': total_loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                }
                if 'mae_loss' in losses:
                    log_dict['train/mae_loss'] = losses['mae_loss'].item()
                if 'contrastive_loss' in losses:
                    log_dict['train/contrastive_loss'] = losses['contrastive_loss'].item()
                if 'forecasting_loss' in losses:
                    log_dict['train/forecasting_loss'] = losses['forecasting_loss'].item()

                self.logger.log(log_dict, step=self.global_step)

            self.global_step += 1

        # Compute epoch averages
        avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}

        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            features = batch['features'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)

            # Forward pass (no masking during validation)
            outputs = self.model(
                features,
                apply_masking=False,
                valid_mask=valid_mask,
            )

            # Compute losses (without masking, just for monitoring)
            losses = self.criterion(outputs, features)
            val_losses.append(losses['total_loss'].item())

        avg_val_loss = np.mean(val_losses)

        return {'total': avg_val_loss}

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Log to wandb
        self.logger.log_artifact(
            artifact_name=f"checkpoint_epoch_{self.current_epoch}",
            artifact_type="model",
            file_path=str(checkpoint_path),
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Checkpoint loaded: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        print("=" * 80)
        print("Starting training...")
        print("=" * 80)

        for epoch in range(self.current_epoch, self.cfg.train.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_losses = self.train_epoch()

            # Record losses
            self.train_losses['total'].append(train_losses['total'])
            if 'mae' in train_losses:
                self.train_losses['mae'].append(train_losses['mae'])
            if 'contrastive' in train_losses:
                self.train_losses['contrastive'].append(train_losses['contrastive'])
            if 'forecasting' in train_losses:
                self.train_losses['forecasting'].append(train_losses['forecasting'])

            # Validate
            if self.val_loader and (epoch + 1) % self.cfg.train.validation.interval == 0:
                val_losses = self.validate()
                self.val_losses['total'].append(val_losses['total'])

                print(f"Epoch {epoch + 1}: Train Loss = {train_losses['total']:.4f}, Val Loss = {val_losses['total']:.4f}")

                # Log to wandb
                self.logger.log({
                    'epoch': epoch + 1,
                    'val/total_loss': val_losses['total'],
                }, step=self.global_step)

                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    if self.cfg.train.checkpoint.save_best:
                        self.save_checkpoint('best_model.pt')

            else:
                print(f"Epoch {epoch + 1}: Train Loss = {train_losses['total']:.4f}")

            # Log epoch metrics
            self.logger.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_losses['total'],
            }, step=self.global_step)

            # Step scheduler
            if self.scheduler and epoch >= self.cfg.train.scheduler.warmup_epochs:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.cfg.train.checkpoint.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        # Save final model
        if self.cfg.train.checkpoint.save_last:
            self.save_checkpoint('last_model.pt')

        # Plot and log loss curves
        loss_curve_img = plot_loss_curves(self.train_losses)
        self.logger.log_image('train/loss_curves', loss_curve_img)

        print("=" * 80)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 80)

        self.logger.finish()


@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
