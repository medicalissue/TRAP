"""
Dataset for loading temporal track sequences.
Supports offline extracted features and ring buffer for online inference.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import deque


class TrackSequenceDataset(Dataset):
    """
    Dataset for loading pre-extracted temporal sequences.
    Each sample is a sequence of object-level features with metadata.
    """

    def __init__(
        self,
        feature_dir: Path,
        split: str = 'train',
        datasets: List[str] = None,
        seq_length: int = 16,
        augment: bool = False,
    ):
        """
        Args:
            feature_dir: Root directory with extracted features
            split: 'train' or 'test'
            datasets: List of dataset names to load
            seq_length: Expected sequence length
            augment: Whether to apply data augmentation
        """
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.seq_length = seq_length
        self.augment = augment

        # Load all sequences
        self.sequences = []
        self.dataset_names = []

        if datasets is None:
            datasets = [d.name for d in self.feature_dir.iterdir() if d.is_dir()]

        for dataset_name in datasets:
            dataset_dir = self.feature_dir / dataset_name / split
            if not dataset_dir.exists():
                print(f"Warning: {dataset_dir} does not exist, skipping...")
                continue

            # Load all feature files
            feature_files = list(dataset_dir.glob("*_features.pt"))
            print(f"Loading {len(feature_files)} files from {dataset_name}/{split}")

            for fpath in feature_files:
                data = torch.load(fpath)
                sequences = data['sequences']

                for seq in sequences:
                    self.sequences.append(seq)
                    self.dataset_names.append(dataset_name)

        print(f"Loaded {len(self.sequences)} sequences from {len(datasets)} datasets")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence sample.

        Returns:
            Dictionary with:
                - features: [T, d_model] sequence features
                - valid_mask: [T] boolean mask
                - track_id: scalar track ID
                - dataset_name: string dataset name
        """
        seq = self.sequences[idx]

        features = seq['features']  # [T, d_model]
        valid_mask = seq.get('valid_mask', torch.ones(features.size(0), dtype=torch.bool))

        # Apply augmentation if enabled
        if self.augment and self.split == 'train':
            features, valid_mask = self.apply_augmentation(features, valid_mask)

        return {
            'features': features,
            'valid_mask': valid_mask,
            'track_id': seq['track_id'],
            'dataset_name': self.dataset_names[idx],
            'frame_range': seq.get('frame_range', (0, self.seq_length - 1)),
        }

    def apply_augmentation(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation to features.

        Args:
            features: [T, d_model]
            valid_mask: [T]

        Returns:
            Augmented features and mask
        """
        # Temporal shift (randomly shift sequence start)
        if torch.rand(1).item() < 0.1:
            shift = torch.randint(-2, 3, (1,)).item()
            features = torch.roll(features, shift, dims=0)
            valid_mask = torch.roll(valid_mask, shift, dims=0)

        # Feature dropout (randomly drop some feature dimensions)
        if torch.rand(1).item() < 0.1:
            dropout_mask = torch.rand(features.size(1)) > 0.1
            features = features * dropout_mask

        # Gaussian noise
        if torch.rand(1).item() < 0.1:
            noise = torch.randn_like(features) * 0.01
            features = features + noise

        return features, valid_mask


class RingBufferSequencer:
    """
    Ring buffer for maintaining temporal sequences during online inference.
    Each track has its own buffer of length T.
    """

    def __init__(self, seq_length: int = 16, d_model: int = 256):
        """
        Args:
            seq_length: Length of temporal sequences
            d_model: Feature dimension
        """
        self.seq_length = seq_length
        self.d_model = d_model

        # Track buffers: track_id -> deque of features
        self.buffers = {}

    def update(self, track_id: int, feature: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Add a new feature for a track and return a sequence if buffer is full.

        Args:
            track_id: Track ID
            feature: [d_model] feature vector

        Returns:
            [T, d_model] sequence if buffer is full, else None
        """
        # Initialize buffer for new tracks
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.seq_length)

        # Add feature to buffer
        self.buffers[track_id].append(feature)

        # Return sequence if buffer is full
        if len(self.buffers[track_id]) == self.seq_length:
            sequence = torch.stack(list(self.buffers[track_id]))  # [T, d_model]
            return sequence

        return None

    def get_partial_sequence(
        self,
        track_id: int,
        pad: bool = True
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get current sequence for a track, even if not full.

        Args:
            track_id: Track ID
            pad: Whether to pad sequences shorter than seq_length

        Returns:
            Tuple of (sequence, valid_mask) if track exists, else None
        """
        if track_id not in self.buffers or len(self.buffers[track_id]) == 0:
            return None

        features = list(self.buffers[track_id])
        seq_len = len(features)

        if pad and seq_len < self.seq_length:
            # Pad with zeros
            padding = [torch.zeros_like(features[0]) for _ in range(self.seq_length - seq_len)]
            features = features + padding

            # Create valid mask
            valid_mask = torch.zeros(self.seq_length, dtype=torch.bool)
            valid_mask[:seq_len] = True
        else:
            valid_mask = torch.ones(seq_len, dtype=torch.bool)

        sequence = torch.stack(features)

        return sequence, valid_mask

    def clear_track(self, track_id: int):
        """Clear buffer for a specific track."""
        if track_id in self.buffers:
            del self.buffers[track_id]

    def clear_all(self):
        """Clear all track buffers."""
        self.buffers.clear()

    def get_active_tracks(self) -> List[int]:
        """Get list of active track IDs."""
        return list(self.buffers.keys())


def collate_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of sequence dictionaries

    Returns:
        Batched dictionary
    """
    # Stack features
    features = torch.stack([item['features'] for item in batch])  # [B, T, d_model]
    valid_masks = torch.stack([item['valid_mask'] for item in batch])  # [B, T]

    # Collect metadata
    track_ids = torch.tensor([item['track_id'] for item in batch])
    dataset_names = [item['dataset_name'] for item in batch]
    frame_ranges = [item['frame_range'] for item in batch]

    return {
        'features': features,
        'valid_mask': valid_masks,
        'track_ids': track_ids,
        'dataset_names': dataset_names,
        'frame_ranges': frame_ranges,
    }


def create_dataloader(
    feature_dir: Path,
    split: str,
    datasets: List[str],
    batch_size: int,
    seq_length: int,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = False,
) -> DataLoader:
    """
    Create DataLoader for training or evaluation.

    Args:
        feature_dir: Root directory with extracted features
        split: 'train' or 'test'
        datasets: List of dataset names
        batch_size: Batch size
        seq_length: Sequence length
        num_workers: Number of workers
        shuffle: Whether to shuffle
        augment: Whether to apply augmentation

    Returns:
        DataLoader
    """
    dataset = TrackSequenceDataset(
        feature_dir=feature_dir,
        split=split,
        datasets=datasets,
        seq_length=seq_length,
        augment=augment,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sequences,
        pin_memory=True,
        drop_last=(split == 'train'),  # Drop last incomplete batch in training
    )

    return dataloader


if __name__ == "__main__":
    # Test dataset and ring buffer
    print("Testing TrackSequenceDataset and RingBufferSequencer...")

    # Create dummy feature directory structure for testing
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy features
        feature_dir = tmpdir / "features"
        dataset_dir = feature_dir / "test_dataset" / "train"
        dataset_dir.mkdir(parents=True)

        # Create dummy sequences
        seq_length = 16
        d_model = 256
        num_tracks = 10

        sequences = []
        for track_id in range(num_tracks):
            for start_idx in range(5):  # 5 sequences per track
                seq = {
                    'track_id': track_id,
                    'features': torch.randn(seq_length, d_model),
                    'frame_range': (start_idx, start_idx + seq_length - 1),
                    'bboxes': np.random.rand(seq_length, 4),
                    'valid_mask': torch.ones(seq_length, dtype=torch.bool),
                }
                sequences.append(seq)

        torch.save({
            'sequences': sequences,
            'metadata': {'num_sequences': len(sequences)},
        }, dataset_dir / "test_video_features.pt")

        # Test dataset
        dataset = TrackSequenceDataset(
            feature_dir=feature_dir,
            split='train',
            datasets=['test_dataset'],
            seq_length=seq_length,
        )

        print(f"Dataset size: {len(dataset)}")
        assert len(dataset) == len(sequences)

        # Test dataloader
        dataloader = create_dataloader(
            feature_dir=feature_dir,
            split='train',
            datasets=['test_dataset'],
            batch_size=8,
            seq_length=seq_length,
            num_workers=0,
            shuffle=True,
        )

        batch = next(iter(dataloader))
        print(f"Batch features shape: {batch['features'].shape}")
        print(f"Batch valid_mask shape: {batch['valid_mask'].shape}")
        assert batch['features'].shape == (8, seq_length, d_model)

    # Test ring buffer
    print("\nTesting RingBufferSequencer...")
    ring_buffer = RingBufferSequencer(seq_length=16, d_model=256)

    # Add features for track 0
    for i in range(20):
        feature = torch.randn(256)
        sequence = ring_buffer.update(track_id=0, feature=feature)

        if i < 15:
            assert sequence is None, f"Buffer should not be full at step {i}"
        else:
            assert sequence is not None, f"Buffer should be full at step {i}"
            assert sequence.shape == (16, 256)

    # Test partial sequence
    ring_buffer.clear_all()
    for i in range(10):
        feature = torch.randn(256)
        ring_buffer.update(track_id=1, feature=feature)

    partial_seq, valid_mask = ring_buffer.get_partial_sequence(track_id=1, pad=True)
    print(f"Partial sequence shape: {partial_seq.shape}")
    print(f"Valid positions: {valid_mask.sum().item()} / {len(valid_mask)}")
    assert partial_seq.shape == (16, 256)
    assert valid_mask.sum().item() == 10

    print("\nAll tests passed!")
