"""
Evaluation script with TAO-compatible metrics.
Evaluates on test datasets and computes pixel, frame, and object-level metrics.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pandas as pd
import json

from models.temporal_mae import TemporalMAE
from models.losses import compute_anomaly_score
from data.trackseq_dataset import create_dataloader
from utils.metrics import evaluate_all_metrics
from utils.evt import select_threshold
from utils.logging import WandbLogger, create_summary_table
from utils.seed import set_seed
from utils.device import resolve_device


class Evaluator:
    """Evaluator for TAO-compatible anomaly detection."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg)

        # Set seed
        set_seed(cfg.seed)

        # Initialize wandb
        self.logger = WandbLogger(
            config=OmegaConf.to_container(cfg, resolve=True),
            enabled=(cfg.wandb.mode != 'disabled')
        )

        # Load model
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

        # Results directory
        self.results_dir = Path(cfg.paths.results)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Evaluator initialized on device: {self.device}")

    def load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        # Build model
        model = TemporalMAE(
            d_model=self.cfg.model.temporal.d_model,
            depth=self.cfg.model.temporal.depth,
            num_heads=self.cfg.model.temporal.num_heads,
            mlp_ratio=self.cfg.model.temporal.mlp_ratio,
            dropout=0.0,  # No dropout during eval
            attn_dropout=0.0,
            mask_ratio=0.0,  # No masking during eval
            use_mae=self.cfg.model.heads.use_mae,
            use_contrastive=self.cfg.model.heads.use_contrastive,
            use_forecasting=False,  # Don't need forecasting for eval
            projection_dim=self.cfg.model.contrastive.projection_dim,
        )

        # Load checkpoint
        checkpoint_name = self.cfg.eval.checkpoint
        if checkpoint_name == 'best':
            checkpoint_path = Path(self.cfg.paths.checkpoints) / 'best_model.pt'
        elif checkpoint_name == 'last':
            checkpoint_path = Path(self.cfg.paths.checkpoints) / 'last_model.pt'
        else:
            checkpoint_path = Path(checkpoint_name)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    @torch.no_grad()
    def compute_sequence_scores(
        self,
        dataloader,
    ) -> Dict[str, np.ndarray]:
        """
        Compute anomaly scores for all sequences in dataloader.

        Returns:
            Dictionary with scores and metadata
        """
        all_recon_errors = []
        all_temp_similarities = []
        all_anomaly_scores = []
        all_track_ids = []
        all_frame_ranges = []
        all_datasets = []

        for batch in tqdm(dataloader, desc="Computing scores"):
            features = batch['features'].to(self.device)  # [B, T, d_model]
            valid_mask = batch['valid_mask'].to(self.device)  # [B, T]

            # Forward pass
            outputs = self.model(
                features,
                apply_masking=False,
                valid_mask=valid_mask,
            )

            # Compute reconstruction error
            if 'reconstruction' in outputs:
                recon = outputs['reconstruction']
                recon_error = ((recon - features) ** 2).mean(dim=-1)  # [B, T]

                # Average over valid positions
                recon_error = (recon_error * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)  # [B]
            else:
                recon_error = torch.zeros(features.size(0), device=self.device)

            # Compute temporal similarity
            # Mean cosine similarity between adjacent frames
            temp_sim = []
            for i in range(features.size(0)):
                valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_indices) > 1:
                    feats = features[i, valid_indices]  # [T_valid, d_model]
                    # Compute cosine similarity between adjacent frames
                    curr = feats[:-1]
                    next_f = feats[1:]
                    sim = torch.cosine_similarity(curr, next_f, dim=-1)  # [T_valid-1]
                    temp_sim.append(sim.mean().item())
                else:
                    temp_sim.append(1.0)  # Default to high similarity if not enough frames

            temp_sim = torch.tensor(temp_sim, device=self.device)

            # Compute anomaly score
            anomaly_score = compute_anomaly_score(
                recon_error,
                temp_sim,
                alpha=self.cfg.eval.scoring.alpha
            )

            # Collect results
            all_recon_errors.append(recon_error.cpu().numpy())
            all_temp_similarities.append(temp_sim.cpu().numpy())
            all_anomaly_scores.append(anomaly_score.cpu().numpy())
            all_track_ids.extend(batch['track_ids'].cpu().numpy())
            all_frame_ranges.extend(batch['frame_ranges'])
            all_datasets.extend(batch['dataset_names'])

        # Concatenate results
        results = {
            'recon_errors': np.concatenate(all_recon_errors),
            'temp_similarities': np.concatenate(all_temp_similarities),
            'anomaly_scores': np.concatenate(all_anomaly_scores),
            'track_ids': np.array(all_track_ids),
            'frame_ranges': all_frame_ranges,
            'datasets': all_datasets,
        }

        return results

    def aggregate_frame_scores(
        self,
        sequence_results: Dict,
        num_frames: int,
    ) -> np.ndarray:
        """
        Aggregate sequence-level scores to frame-level.

        Args:
            sequence_results: Results from compute_sequence_scores
            num_frames: Total number of frames in video

        Returns:
            [num_frames] frame-level anomaly scores
        """
        frame_scores = np.zeros(num_frames)
        frame_counts = np.zeros(num_frames)

        # Aggregate scores by frame
        for score, frame_range in zip(sequence_results['anomaly_scores'], sequence_results['frame_ranges']):
            start, end = frame_range
            for frame_idx in range(start, end + 1):
                if frame_idx < num_frames:
                    if self.cfg.eval.scoring.aggregation == 'max':
                        frame_scores[frame_idx] = max(frame_scores[frame_idx], score)
                    elif self.cfg.eval.scoring.aggregation == 'mean':
                        frame_scores[frame_idx] += score
                        frame_counts[frame_idx] += 1

        # Average for mean aggregation
        if self.cfg.eval.scoring.aggregation == 'mean':
            frame_scores = np.divide(
                frame_scores,
                frame_counts,
                where=frame_counts > 0,
                out=frame_scores
            )

        # Apply EWMA smoothing
        if self.cfg.eval.scoring.beta < 1.0:
            frame_scores = self.apply_ewma(frame_scores, self.cfg.eval.scoring.beta)

        # Normalize scores
        if self.cfg.eval.scoring.normalize_scores:
            if frame_scores.max() > frame_scores.min():
                frame_scores = (frame_scores - frame_scores.min()) / (frame_scores.max() - frame_scores.min())

        return frame_scores

    def apply_ewma(self, scores: np.ndarray, beta: float) -> np.ndarray:
        """Apply Exponentially Weighted Moving Average smoothing."""
        smoothed = np.zeros_like(scores)
        smoothed[0] = scores[0]

        for i in range(1, len(scores)):
            smoothed[i] = beta * smoothed[i-1] + (1 - beta) * scores[i]

        return smoothed

    def evaluate_dataset(self, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate on a single dataset.

        Args:
            dataset_name: Name of dataset

        Returns:
            Dictionary with metrics
        """
        print(f"\n{'='*80}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*80}")

        # Create dataloader for test set
        test_loader = create_dataloader(
            feature_dir=Path(self.cfg.data.features_root),
            split='test',
            datasets=[dataset_name],
            batch_size=self.cfg.train.batch_size,
            seq_length=self.cfg.data.sequence.T,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            augment=False,
        )

        # Compute scores
        sequence_results = self.compute_sequence_scores(test_loader)

        # TODO: Load ground truth labels
        # For now, create dummy labels
        num_frames = max([fr[1] for fr in sequence_results['frame_ranges']]) + 1
        frame_labels = np.zeros(num_frames)  # Dummy labels

        # Aggregate to frame level
        frame_scores = self.aggregate_frame_scores(sequence_results, num_frames)

        # Select threshold
        if self.cfg.eval.threshold.method == 'evt':
            threshold = select_threshold(
                sequence_results['anomaly_scores'],
                method='evt',
                tail_size=self.cfg.eval.threshold.evt_tail_size,
            )
        elif self.cfg.eval.threshold.method == 'fixed':
            # Use dataset-specific threshold if available
            threshold = self.cfg.eval.threshold.get(dataset_name, self.cfg.eval.threshold.fixed_value)
        elif self.cfg.eval.threshold.method == 'percentile':
            threshold = select_threshold(
                sequence_results['anomaly_scores'],
                method='percentile',
                percentile=self.cfg.eval.threshold.percentile,
            )
        else:
            threshold = 1.5

        print(f"Threshold: {threshold:.4f}")

        # Compute metrics
        predictions = {
            'frame_scores': frame_scores,
            # 'pixel_scores': ...,  # Would need to compute from masks
            # 'pred_boxes': ...,
            # 'pred_scores': ...,
            # 'track_ids': ...,
            # 'track_scores': ...,
        }

        ground_truth = {
            'frame_labels': frame_labels,
            # 'pixel_masks': ...,
        }

        metrics = evaluate_all_metrics(predictions, ground_truth, threshold)

        # Save predictions
        if self.cfg.eval.output.save_per_frame_csv:
            csv_dir = Path(self.cfg.eval.output.csv_dir)
            csv_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame({
                'frame_idx': np.arange(len(frame_scores)),
                'anomaly_score': frame_scores,
                'label': frame_labels,
                'is_anomaly': frame_scores > threshold,
            })
            df.to_csv(csv_dir / f"{dataset_name}_frame_predictions.csv", index=False)

        if self.cfg.eval.output.save_per_object_csv:
            csv_dir = Path(self.cfg.eval.output.csv_dir)
            csv_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame({
                'track_id': sequence_results['track_ids'],
                'recon_error': sequence_results['recon_errors'],
                'temp_similarity': sequence_results['temp_similarities'],
                'anomaly_score': sequence_results['anomaly_scores'],
                'dataset': sequence_results['datasets'],
            })
            df.to_csv(csv_dir / f"{dataset_name}_object_predictions.csv", index=False)

        return metrics

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Evaluate on all datasets."""
        all_results = {}

        for dataset_name in self.cfg.data.datasets:
            try:
                metrics = self.evaluate_dataset(dataset_name)
                all_results[dataset_name] = metrics

                # Log to wandb
                wandb_metrics = {f"{dataset_name}/{k}": v for k, v in metrics.items()}
                self.logger.log(wandb_metrics)

            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                continue

        # Create summary table
        summary_table = create_summary_table(all_results)
        print(f"\n{'='*80}")
        print("Evaluation Summary")
        print(f"{'='*80}")
        print(summary_table)

        # Save results
        results_file = self.results_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        # Compare with baselines
        self.compare_with_baselines(all_results)

        self.logger.finish()

        return all_results

    def compare_with_baselines(self, results: Dict[str, Dict[str, float]]):
        """Compare results with TAO paper baselines."""
        print(f"\n{'='*80}")
        print("Comparison with TAO Baselines")
        print(f"{'='*80}")

        for dataset_name in results.keys():
            if dataset_name in self.cfg.eval.baselines:
                print(f"\n{dataset_name}:")
                baselines = self.cfg.eval.baselines[dataset_name]

                # Print baselines
                print("  Baselines:")
                for method, scores in baselines.items():
                    print(f"    {method}: Frame-AUC={scores.get('frame_auc', 'N/A')}, Pixel-AUC={scores.get('pixel_auc', 'N/A')}")

                # Print ours
                our_results = results[dataset_name]
                print("  Ours:")
                print(f"    Temporal-MAE: Frame-AUC={our_results.get('frame_auc', 'N/A'):.2f}, Pixel-AUC={our_results.get('pixel_auroc', 'N/A')}")


@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """Main evaluation function."""
    print(OmegaConf.to_yaml(cfg))

    evaluator = Evaluator(cfg)
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
