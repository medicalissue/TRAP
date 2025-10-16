"""
Offline feature extraction using YOLOv8 (frozen) + ByteTrack.
Extracts object-level temporal sequences and saves as .pt shards.
"""

import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    from torchvision.ops import RoIAlign
except ImportError:
    print("Please install ultralytics and torchvision:")
    print("pip install ultralytics torchvision")
    sys.exit(1)

from models.pooling import build_pooling


class FeatureExtractor:
    """Extract object-level features from videos using YOLO+ByteTrack."""

    def __init__(self, cfg: DictConfig, device: str = 'cuda'):
        self.cfg = cfg
        self.device = device

        # Load YOLO model with ByteTrack
        print(f"Loading YOLO model: {cfg.model.yolo.weights}")
        self.model = YOLO(cfg.model.yolo.weights)
        self.model.to(device)

        # Freeze YOLO backbone
        if cfg.model.yolo.freeze_backbone:
            for param in self.model.model.parameters():
                param.requires_grad = False
            print("YOLO backbone frozen")

        # Get FPN feature dimensions
        self.fpn_layers = cfg.model.yolo.fpn_layers
        self.roi_size = cfg.model.yolo.roi_size

        # RoIAlign for extracting features
        self.roi_align = RoIAlign(
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
        )

        # Feature pooling module
        # Infer input channels from YOLO model
        self.pooling_mode = cfg.model.pooling.mode
        self.d_model = cfg.model.pooling.d_model

        # Initialize pooling (will be created after extracting first features)
        self.pooling = None

        # Tracking parameters
        self.tracker_config = cfg.model.yolo.tracker
        self.conf_threshold = cfg.model.yolo.conf_threshold
        self.iou_threshold = cfg.model.yolo.iou_threshold

        # Sequence parameters
        self.seq_length = cfg.data.sequence.T
        self.seq_stride = cfg.data.sequence.stride

        # Storage for track sequences
        self.track_buffers = defaultdict(list)  # track_id -> list of features

    def register_hooks(self):
        """Register forward hooks on FPN layers to extract features."""
        self.feature_maps = {}

        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook

        # Register hooks on specified FPN layers
        for idx in self.fpn_layers:
            try:
                layer = self.model.model.model[idx]
                layer.register_forward_hook(get_activation(f'fpn_{idx}'))
            except (IndexError, AttributeError) as e:
                print(f"Warning: Could not register hook on layer {idx}: {e}")

    def extract_frame_features(
        self,
        frame: np.ndarray,
        tracked_results
    ) -> List[Tuple[int, torch.Tensor, np.ndarray]]:
        """
        Extract features for all tracked objects in a frame.

        Args:
            frame: [H, W, 3] RGB frame
            tracked_results: Results from YOLO tracking

        Returns:
            List of (track_id, feature, bbox) tuples
        """
        features_list = []

        if tracked_results.boxes is None or len(tracked_results.boxes) == 0:
            return features_list

        # Get boxes and track IDs
        boxes = tracked_results.boxes.xyxy.cpu().numpy()  # [N, 4]
        if tracked_results.boxes.id is not None:
            track_ids = tracked_results.boxes.id.cpu().numpy().astype(int)
        else:
            # If no track IDs, use sequential IDs
            track_ids = np.arange(len(boxes))

        # Extract features from FPN layers
        # Use the highest resolution FPN layer
        fpn_key = f'fpn_{self.fpn_layers[0]}'
        if fpn_key not in self.feature_maps:
            return features_list

        feature_map = self.feature_maps[fpn_key]  # [1, C, H, W]
        B, C, fH, fW = feature_map.shape

        # Initialize pooling module if not done yet
        if self.pooling is None:
            self.pooling = build_pooling(
                mode=self.pooling_mode,
                in_channels=C,
                d_model=self.d_model,
                num_heads=self.cfg.model.pooling.attn_heads,
                dropout=self.cfg.model.pooling.attn_dropout,
            ).to(self.device)
            print(f"Initialized {self.pooling_mode} pooling: {C} -> {self.d_model}")

        # Get frame dimensions
        imgH, imgW = frame.shape[:2]

        # Scale boxes to feature map coordinates
        scale_x = fW / imgW
        scale_y = fH / imgH

        # Extract RoI features for each box
        for track_id, box in zip(track_ids, boxes):
            x1, y1, x2, y2 = box

            # Scale to feature map coordinates
            x1_f = x1 * scale_x
            y1_f = y1 * scale_y
            x2_f = x2 * scale_x
            y2_f = y2 * scale_y

            # Create RoI tensor [1, 5] format: [batch_idx, x1, y1, x2, y2]
            roi = torch.tensor([[0, x1_f, y1_f, x2_f, y2_f]], device=self.device)

            # Extract RoI features
            roi_features = self.roi_align(feature_map, roi)  # [1, C, roi_size, roi_size]

            # Pool to d_model dimension
            with torch.no_grad():
                pooled_features = self.pooling(roi_features)  # [1, d_model]

            features_list.append((
                track_id,
                pooled_features.squeeze(0).cpu(),  # [d_model]
                box,
            ))

        return features_list

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        dataset_name: str,
    ) -> Dict:
        """
        Process a single video and extract track sequences.

        Args:
            video_path: Path to video file
            output_dir: Directory to save features
            dataset_name: Dataset name

        Returns:
            Dictionary with extraction metadata
        """
        print(f"Processing: {video_path.name}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Reset track buffers for this video
        self.track_buffers.clear()
        frame_idx = 0

        # Process frames
        pbar = tqdm(total=total_frames, desc=video_path.stem)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLO tracking
            results = self.model.track(
                frame_rgb,
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            # Extract features for tracked objects
            frame_features = self.extract_frame_features(frame_rgb, results[0])

            # Update track buffers
            for track_id, feature, bbox in frame_features:
                self.track_buffers[track_id].append({
                    'frame_idx': frame_idx,
                    'feature': feature,
                    'bbox': bbox,
                })

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        # Build sequences from track buffers
        sequences = self.build_sequences()

        # Save sequences
        output_path = output_dir / f"{video_path.stem}_features.pt"
        metadata = {
            'dataset': dataset_name,
            'video_name': video_path.name,
            'total_frames': total_frames,
            'fps': fps,
            'num_tracks': len(self.track_buffers),
            'num_sequences': len(sequences),
        }

        torch.save({
            'sequences': sequences,
            'metadata': metadata,
        }, output_path)

        print(f"Saved {len(sequences)} sequences to {output_path}")

        return metadata

    def build_sequences(self) -> List[Dict]:
        """
        Build temporal sequences from track buffers using sliding window.

        Returns:
            List of sequence dictionaries
        """
        sequences = []

        for track_id, track_data in self.track_buffers.items():
            if len(track_data) < self.seq_length:
                continue  # Skip short tracks

            # Sort by frame index
            track_data = sorted(track_data, key=lambda x: x['frame_idx'])

            # Extract features and metadata
            features = torch.stack([d['feature'] for d in track_data])  # [T, d_model]
            frame_indices = [d['frame_idx'] for d in track_data]
            bboxes = np.array([d['bbox'] for d in track_data])

            # Sliding window to create sequences
            for start_idx in range(0, len(features) - self.seq_length + 1, self.seq_stride):
                end_idx = start_idx + self.seq_length

                seq_dict = {
                    'track_id': track_id,
                    'features': features[start_idx:end_idx],  # [T, d_model]
                    'frame_range': (frame_indices[start_idx], frame_indices[end_idx - 1]),
                    'bboxes': bboxes[start_idx:end_idx],  # [T, 4]
                    'valid_mask': torch.ones(self.seq_length, dtype=torch.bool),
                }

                sequences.append(seq_dict)

        return sequences


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """Main extraction function."""
    print("=" * 80)
    print("Offline Feature Extraction: YOLOv8 + ByteTrack")
    print("=" * 80)

    # Setup device
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(cfg.paths.features_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = FeatureExtractor(cfg, device)
    extractor.register_hooks()

    # Process datasets
    all_metadata = []

    for dataset_name in cfg.data.datasets:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")

        # Get dataset config
        dataset_cfg = cfg.data.get(dataset_name)
        if dataset_cfg is None:
            print(f"No configuration found for {dataset_name}, skipping...")
            continue

        dataset_root = Path(dataset_cfg.root)
        if not dataset_root.exists():
            print(f"Dataset root not found: {dataset_root}, skipping...")
            continue

        # Process training videos (normal only)
        train_dir = dataset_root / dataset_cfg.train_split
        if train_dir.exists():
            print(f"\nProcessing training videos from: {train_dir}")

            # Find all video files
            video_files = list(train_dir.rglob(f"*{dataset_cfg.video_format.replace('*', '')}"))
            if not video_files:
                # Try finding as image sequences
                video_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
                print(f"Found {len(video_dirs)} video directories")
            else:
                print(f"Found {len(video_files)} video files")

                for video_path in video_files:
                    metadata = extractor.process_video(
                        video_path,
                        output_dir / dataset_name / 'train',
                        dataset_name
                    )
                    all_metadata.append(metadata)

        # Process test videos
        test_dir = dataset_root / dataset_cfg.test_split
        if test_dir.exists():
            print(f"\nProcessing test videos from: {test_dir}")

            video_files = list(test_dir.rglob(f"*{dataset_cfg.video_format.replace('*', '')}"))
            if video_files:
                print(f"Found {len(video_files)} video files")

                for video_path in video_files:
                    metadata = extractor.process_video(
                        video_path,
                        output_dir / dataset_name / 'test',
                        dataset_name
                    )
                    all_metadata.append(metadata)

    # Save overall metadata
    summary_path = output_dir / "extraction_summary.pt"
    torch.save({
        'config': dict(cfg),
        'all_metadata': all_metadata,
    }, summary_path)

    print(f"\n{'='*80}")
    print(f"Extraction complete! Processed {len(all_metadata)} videos.")
    print(f"Features saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
