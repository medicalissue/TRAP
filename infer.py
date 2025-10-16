"""
Online inference script with sliding window and visualization.
Processes video frame-by-frame with real-time anomaly detection.
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    from torchvision.ops import RoIAlign
except ImportError:
    print("Please install ultralytics and torchvision:")
    print("pip install ultralytics torchvision")
    sys.exit(1)

from models.temporal_mae import TemporalMAE
from models.pooling import build_pooling
from models.losses import compute_anomaly_score
from data.trackseq_dataset import RingBufferSequencer
from utils.seed import set_seed


class OnlineInference:
    """Online inference with YOLOv8+ByteTrack and Temporal MAE."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() else 'cpu'

        # Set seed
        set_seed(cfg.seed)

        # Load YOLO model
        print(f"Loading YOLO model: {cfg.model.yolo.weights}")
        self.yolo_model = YOLO(cfg.model.yolo.weights)
        self.yolo_model.to(self.device)

        # Freeze YOLO
        if cfg.model.yolo.freeze_backbone:
            for param in self.yolo_model.model.parameters():
                param.requires_grad = False

        # Feature extraction setup
        self.fpn_layers = cfg.model.yolo.fpn_layers
        self.roi_size = cfg.model.yolo.roi_size
        self.feature_maps = {}

        # Register hooks
        self.register_hooks()

        # RoIAlign
        self.roi_align = RoIAlign(
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
        )

        # Pooling module
        self.pooling = None
        self.d_model = cfg.model.pooling.d_model

        # Load Temporal MAE
        self.temporal_model = self.load_temporal_model()
        self.temporal_model.to(self.device)
        self.temporal_model.eval()

        # Ring buffer for maintaining sequences
        self.ring_buffer = RingBufferSequencer(
            seq_length=cfg.data.sequence.T,
            d_model=self.d_model,
        )

        # Tracking configuration
        self.tracker_config = cfg.model.yolo.tracker
        self.conf_threshold = cfg.model.yolo.conf_threshold

        # Scoring configuration
        self.alpha = cfg.eval.scoring.alpha
        self.beta = cfg.eval.scoring.beta

        # Track histories for visualization
        self.track_histories = {}  # track_id -> {'scores': [], 'boxes': []}

        print(f"Online inference initialized on device: {self.device}")

    def register_hooks(self):
        """Register forward hooks on FPN layers."""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook

        for idx in self.fpn_layers:
            try:
                layer = self.yolo_model.model.model[idx]
                layer.register_forward_hook(get_activation(f'fpn_{idx}'))
            except (IndexError, AttributeError) as e:
                print(f"Warning: Could not register hook on layer {idx}: {e}")

    def load_temporal_model(self) -> torch.nn.Module:
        """Load trained Temporal MAE model."""
        model = TemporalMAE(
            d_model=self.cfg.model.temporal.d_model,
            depth=self.cfg.model.temporal.depth,
            num_heads=self.cfg.model.temporal.num_heads,
            mlp_ratio=self.cfg.model.temporal.mlp_ratio,
            dropout=0.0,
            attn_dropout=0.0,
            mask_ratio=0.0,
            use_mae=self.cfg.model.heads.use_mae,
            use_contrastive=self.cfg.model.heads.use_contrastive,
            use_forecasting=False,
            projection_dim=self.cfg.model.contrastive.projection_dim,
        )

        # Load checkpoint
        checkpoint_path = Path(self.cfg.paths.checkpoints) / 'best_model.pt'
        if not checkpoint_path.exists():
            checkpoint_path = Path(self.cfg.paths.checkpoints) / 'last_model.pt'

        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Warning: No checkpoint found, using untrained model")

        return model

    def extract_object_features(
        self,
        frame: np.ndarray,
        tracked_results
    ) -> List[Tuple[int, torch.Tensor, np.ndarray]]:
        """Extract features for tracked objects in a frame."""
        features_list = []

        if tracked_results.boxes is None or len(tracked_results.boxes) == 0:
            return features_list

        # Get boxes and track IDs
        boxes = tracked_results.boxes.xyxy.cpu().numpy()
        if tracked_results.boxes.id is not None:
            track_ids = tracked_results.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(boxes))

        # Get FPN features
        fpn_key = f'fpn_{self.fpn_layers[0]}'
        if fpn_key not in self.feature_maps:
            return features_list

        feature_map = self.feature_maps[fpn_key]
        B, C, fH, fW = feature_map.shape

        # Initialize pooling if needed
        if self.pooling is None:
            self.pooling = build_pooling(
                mode=self.cfg.model.pooling.mode,
                in_channels=C,
                d_model=self.d_model,
                num_heads=self.cfg.model.pooling.attn_heads,
                dropout=self.cfg.model.pooling.attn_dropout,
            ).to(self.device)
            print(f"Initialized {self.cfg.model.pooling.mode} pooling: {C} -> {self.d_model}")

        # Extract features for each box
        imgH, imgW = frame.shape[:2]
        scale_x = fW / imgW
        scale_y = fH / imgH

        for track_id, box in zip(track_ids, boxes):
            x1, y1, x2, y2 = box
            x1_f = x1 * scale_x
            y1_f = y1 * scale_y
            x2_f = x2 * scale_x
            y2_f = y2 * scale_y

            roi = torch.tensor([[0, x1_f, y1_f, x2_f, y2_f]], device=self.device)
            roi_features = self.roi_align(feature_map, roi)

            with torch.no_grad():
                pooled = self.pooling(roi_features).squeeze(0).cpu()

            features_list.append((track_id, pooled, box))

        return features_list

    @torch.no_grad()
    def compute_anomaly_score_for_sequence(
        self,
        sequence: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute anomaly score for a sequence.

        Returns:
            Tuple of (anomaly_score, recon_error, temp_similarity)
        """
        # Add batch dimension
        sequence = sequence.unsqueeze(0).to(self.device)  # [1, T, d_model]
        valid_mask = valid_mask.unsqueeze(0).to(self.device)  # [1, T]

        # Forward pass
        outputs = self.temporal_model(
            sequence,
            apply_masking=False,
            valid_mask=valid_mask,
        )

        # Reconstruction error
        if 'reconstruction' in outputs:
            recon = outputs['reconstruction']
            recon_error = ((recon - sequence) ** 2).mean(dim=-1)  # [1, T]
            recon_error = (recon_error * valid_mask).sum() / valid_mask.sum()
            recon_error = recon_error.item()
        else:
            recon_error = 0.0

        # Temporal similarity
        valid_indices = valid_mask[0].nonzero(as_tuple=True)[0]
        if len(valid_indices) > 1:
            feats = sequence[0, valid_indices]
            curr = feats[:-1]
            next_f = feats[1:]
            sim = torch.cosine_similarity(curr, next_f, dim=-1)
            temp_similarity = sim.mean().item()
        else:
            temp_similarity = 1.0

        # Compute anomaly score
        anomaly_score = compute_anomaly_score(
            torch.tensor([recon_error]),
            torch.tensor([temp_similarity]),
            alpha=self.alpha
        ).item()

        return anomaly_score, recon_error, temp_similarity

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> Dict[int, Dict]:
        """
        Process a single frame and return anomaly scores for each track.

        Returns:
            Dictionary mapping track_id to score info
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO tracking
        results = self.yolo_model.track(
            frame_rgb,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            verbose=False,
        )

        # Extract features
        object_features = self.extract_object_features(frame_rgb, results[0])

        # Process each tracked object
        track_scores = {}

        for track_id, feature, box in object_features:
            # Update ring buffer
            sequence = self.ring_buffer.update(track_id, feature)

            if sequence is not None:
                # Buffer is full, compute anomaly score
                valid_mask = torch.ones(sequence.size(0), dtype=torch.bool)
                anomaly_score, recon_error, temp_sim = self.compute_anomaly_score_for_sequence(
                    sequence, valid_mask
                )
            else:
                # Buffer not full yet, use partial sequence
                result = self.ring_buffer.get_partial_sequence(track_id, pad=True)
                if result is not None:
                    sequence, valid_mask = result
                    anomaly_score, recon_error, temp_sim = self.compute_anomaly_score_for_sequence(
                        sequence, valid_mask
                    )
                else:
                    anomaly_score, recon_error, temp_sim = 0.0, 0.0, 1.0

            # Store score info
            track_scores[track_id] = {
                'anomaly_score': anomaly_score,
                'recon_error': recon_error,
                'temp_similarity': temp_sim,
                'box': box,
            }

            # Update track history
            if track_id not in self.track_histories:
                self.track_histories[track_id] = {
                    'scores': [],
                    'boxes': [],
                    'frames': [],
                }

            self.track_histories[track_id]['scores'].append(anomaly_score)
            self.track_histories[track_id]['boxes'].append(box)
            self.track_histories[track_id]['frames'].append(frame_idx)

        return track_scores

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        visualize: bool = True
    ) -> Dict:
        """
        Process entire video.

        Args:
            video_path: Path to input video
            output_path: Path to save output video (if visualize=True)
            visualize: Whether to create annotated video

        Returns:
            Dictionary with all results
        """
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output video writer
        if visualize and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None

        # Reset state
        self.ring_buffer.clear_all()
        self.track_histories.clear()

        # Process frames
        frame_results = []

        pbar = tqdm(total=total_frames, desc="Processing frames")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            track_scores = self.process_frame(frame, frame_idx)

            # Aggregate to frame-level score
            if track_scores:
                frame_score = max([info['anomaly_score'] for info in track_scores.values()])
            else:
                frame_score = 0.0

            frame_results.append({
                'frame_idx': frame_idx,
                'frame_score': frame_score,
                'track_scores': track_scores,
            })

            # Visualize
            if visualize:
                annotated_frame = self.draw_annotations(frame.copy(), track_scores)
                if out:
                    out.write(annotated_frame)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        if out:
            out.release()

        pbar.close()

        print(f"Processed {frame_idx} frames")
        if output_path:
            print(f"Saved annotated video to: {output_path}")

        return {
            'frame_results': frame_results,
            'track_histories': self.track_histories,
            'video_info': {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
            }
        }

    def draw_annotations(
        self,
        frame: np.ndarray,
        track_scores: Dict[int, Dict]
    ) -> np.ndarray:
        """Draw bounding boxes with color-coded anomaly scores."""
        for track_id, info in track_scores.items():
            score = info['anomaly_score']
            box = info['box'].astype(int)

            # Color based on score (green -> yellow -> red)
            if score < 0.3:
                color = (0, 255, 0)  # Green (normal)
            elif score < 0.6:
                color = (0, 255, 255)  # Yellow (suspicious)
            else:
                color = (0, 0, 255)  # Red (anomaly)

            # Draw box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Draw label
            label = f"ID{track_id}: {score:.3f}"
            cv2.putText(
                frame, label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

        return frame


@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """Main inference function."""
    print("=" * 80)
    print("Online Inference: Temporal MAE + YOLOv8")
    print("=" * 80)

    # Get source video
    if 'source' not in cfg:
        print("Error: Please specify source video with: source=path/to/video.mp4")
        return

    video_path = cfg.source

    # Create inference object
    inferencer = OnlineInference(cfg)

    # Output path
    output_dir = Path(cfg.paths.results) / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    output_path = output_dir / f"{video_name}_annotated.mp4"

    # Process video
    results = inferencer.process_video(
        video_path=video_path,
        output_path=str(output_path),
        visualize=cfg.get('viz', {}).get('enable', True)
    )

    # Save results
    results_path = output_dir / f"{video_name}_results.pt"
    torch.save(results, results_path)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
