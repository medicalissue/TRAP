# Temporal MAE for Object-Level Video Anomaly Detection

A modular PyTorch implementation for object-level video anomaly and OOD detection, combining **YOLOv8 + ByteTrack + Temporal MAE with Contrastive Learning**. This codebase reproduces an evaluation protocol compatible with the CVPR 2025 paper **"Track Any Anomalous Object (TAO)"**.

## Features

- 🎯 **Object-Level Detection**: Track and detect anomalies at the object level using YOLOv8+ByteTrack
- 🧠 **Temporal MAE**: Transformer-based masked autoencoder for temporal sequence modeling
- 🔄 **Contrastive Learning**: InfoNCE loss for temporal consistency
- 📊 **TAO-Compatible Metrics**: Pixel-AUROC, RBDC, TBDC, and more
- ⚙️ **Hydra Configuration**: All settings managed with Hydra for easy experimentation
- 📈 **Weights & Biases**: Complete integration for experiment tracking and visualization
- 🎬 **Online Inference**: Real-time processing with sliding window buffer

## Architecture

```
Pipeline Stages:
1. Feature Extraction: Frozen YOLOv8 + ByteTrack
2. Object-Level Pooling: GAP | Attention | Hybrid
3. Temporal MAE: Encoder-only transformer with masking
4. Multi-Head Outputs: MAE reconstruction + Contrastive learning + Forecasting (optional)
5. Anomaly Scoring: α × reconstruction_error + (1-α) × (1 - temporal_similarity)
```

## Installation

```bash
# Create conda environment
conda create -n temporal_mae python=3.9
conda activate temporal_mae

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install ultralytics hydra-core wandb scipy scikit-learn pandas matplotlib opencv-python tqdm

# Clone and setup
cd temporal_mae
```

## Project Structure

```
project/
├── configs/              # Hydra configuration files
│   ├── main.yaml        # Main config
│   ├── model.yaml       # Model architecture config
│   ├── data.yaml        # Dataset config
│   ├── train.yaml       # Training config
│   └── eval.yaml        # Evaluation config
├── models/              # Model implementations
│   ├── temporal_mae.py  # Temporal MAE model
│   ├── pooling.py       # Feature pooling modules
│   ├── projection_heads.py  # Task-specific heads
│   └── losses.py        # Loss functions
├── data/                # Dataset and data loading
│   └── trackseq_dataset.py
├── extract/             # Feature extraction
│   └── offline_extract.py
├── utils/               # Utilities
│   ├── metrics.py       # TAO metrics (RBDC, TBDC, etc.)
│   ├── evt.py          # Extreme Value Theory thresholding
│   ├── logging.py      # Wandb logging
│   └── seed.py         # Reproducibility
├── viz/                 # Visualization
│   └── visualizer.py
├── train.py            # Training script
├── eval.py             # Evaluation script
├── infer.py            # Online inference script
└── README.md
```

## Quick Start

### 1. Prepare Data

Organize your datasets as follows:

```
data/
├── shanghaitech/
│   ├── training/frames/
│   └── testing/frames/
├── UCSD_Ped2/
│   ├── Train/
│   └── Test/
└── Avenue/
    ├── training_videos/
    └── testing_videos/
```

### 2. Extract Features

Extract object-level features using frozen YOLOv8 + ByteTrack:

```bash
python extract/offline_extract.py \
    data.root=data/shanghaitech \
    model.yolo.weights=yolov8s.pt \
    model.yolo.freeze_backbone=True \
    model.pooling.mode=hybrid \
    data.sequence.T=16 \
    data.sequence.stride=1
```

**Key Parameters:**
- `model.pooling.mode`: `gap` | `attn` | `hybrid`
- `data.sequence.T`: Temporal sequence length (default: 16)
- `data.sequence.stride`: Sliding window stride (default: 1)

### 3. Train Model

Train Temporal MAE on normal-only sequences:

```bash
python train.py \
    data.datasets=[shanghaitech,ucsd_ped2] \
    model.pooling.mode=hybrid \
    model.yolo.weights=yolov8s.pt \
    model.yolo.freeze_backbone=True \
    train.epochs=50 \
    train.batch_size=64 \
    train.optimizer.lr=1e-4 \
    train.losses.lambda_mae=1.0 \
    train.losses.lambda_cl=0.3 \
    wandb.project=temporal_mae_ood \
    wandb.name=hybrid_pooling_exp1
```

**Training Configuration:**
- Training uses **normal videos only** (self-supervised)
- MAE reconstruction loss + InfoNCE contrastive loss
- Optional forecasting head (disabled by default)
- Checkpoints saved to `checkpoints/`

### 4. Evaluate

Evaluate on test datasets with TAO metrics:

```bash
python eval.py \
    data.datasets=[shanghaitech,avenue,ucsd_ped2] \
    eval.checkpoint=best \
    eval.threshold.method=evt \
    eval.scoring.alpha=0.5 \
    eval.metrics=[pixel_auroc,rbdc,tbdc,frame_auc] \
    wandb.project=temporal_mae_ood
```

**Evaluation Metrics:**
- **Pixel-level**: AUROC, AP, AUPRO, F1
- **Object-level**: RBDC (Region-Based Detection Criterion), TBDC (Track-Based Detection Criterion)
- **Frame-level**: AUC, PR-AUC
- **Performance**: Latency, FPS

### 5. Online Inference

Run online inference on a new video:

```bash
python infer.py \
    source=demo.mp4 \
    model.yolo.weights=yolov8s.pt \
    model.pooling.mode=hybrid \
    eval.scoring.alpha=0.5 \
    viz.enable=true \
    wandb.project=temporal_mae_ood
```

Output:
- Annotated video with color-coded bounding boxes
- Per-frame and per-object anomaly scores
- Results saved to `results/videos/`

## Configuration

All configurations are managed with Hydra. Here are key configuration groups:

### Model Configuration (`configs/model.yaml`)

```yaml
yolo:
  weights: yolov8s.pt          # YOLOv8 variant
  freeze_backbone: true         # Freeze YOLO weights
  tracker: bytetrack.yaml       # Built-in ByteTrack

pooling:
  mode: hybrid                  # gap | attn | hybrid
  d_model: 256                  # Feature dimension

temporal:
  d_model: 256
  depth: 6                      # Transformer depth
  num_heads: 8
  mask_ratio: 0.75              # MAE masking ratio
  seq_length: 16                # Temporal window size

contrastive:
  temperature: 0.07
  projection_dim: 128
```

### Training Configuration (`configs/train.yaml`)

```yaml
epochs: 50
batch_size: 64
optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 1e-4

losses:
  lambda_mae: 1.0               # MAE reconstruction weight
  lambda_cl: 0.3                # Contrastive learning weight
  lambda_forecast: 0.1          # Forecasting weight (if enabled)

scheduler:
  name: cosine
  warmup_epochs: 5
  min_lr: 1e-6
```

### Evaluation Configuration (`configs/eval.yaml`)

```yaml
scoring:
  alpha: 0.5                    # Balance: recon vs temporal similarity
  beta: 0.9                     # EWMA smoothing
  aggregation: max              # Frame-level aggregation: max | mean

threshold:
  method: evt                   # evt | fixed | percentile
  ucsd_ped2: 1.5               # Dataset-specific thresholds
  shanghaitech: 1.6

metrics:
  - pixel_auroc
  - pixel_ap
  - rbdc
  - tbdc
  - frame_auc
```

## Supported Datasets

- ✅ **ShanghaiTech Campus** (training + testing)
- ✅ **UCSD Ped2** (training + testing)
- ✅ **Avenue** (training + testing)
- 🔧 **StreetScene** (testing only)
- 🔧 **UBnormal** (testing only)

## TAO Evaluation Protocol

This codebase implements the TAO paper's evaluation protocol:

### Object-Level Metrics

**RBDC (Region-Based Detection Criterion)**:
- Measures localization accuracy of predicted bounding boxes
- IoU-based matching with ground truth anomaly regions
- Computes precision, recall, and F1

**TBDC (Track-Based Detection Criterion)**:
- Measures temporal consistency of detections
- Tracks must match anomaly segments for ≥ m frames within window k
- Default: k=5, m=3

### Threshold Calibration

Following TAO paper recommendations:
- UCSD Ped2: τ = 1.5
- ShanghaiTech: τ = 1.6
- Avenue: τ = 1.5

Auto-calibration via Extreme Value Theory (EVT) also supported.

## Experiment Tracking

All experiments are logged to Weights & Biases:

```python
# Logged metrics
- Training: loss curves, learning rate, gradients (optional)
- Validation: loss, sample reconstructions
- Evaluation: all TAO metrics, comparison with baselines
- Visualizations: attention maps, timelines, galleries

# Logged artifacts
- Model checkpoints
- Configuration files
- Prediction CSVs
- Annotated videos
```

## Visualization

The codebase provides extensive visualization tools:

### 1. Annotated Videos
Color-coded bounding boxes (green → yellow → red) based on anomaly scores.

### 2. Track Timelines
Per-track anomaly score evolution over time.

### 3. Frame Score Plots
Frame-level anomaly scores with top-K highlighting.

### 4. Anomaly Galleries
Grid of most anomalous frames.

### 5. HTML Reports
Comprehensive summary with metrics and statistics.

## Ablation Studies

Enable ablation studies in eval config:

```yaml
ablation:
  enabled: true
  studies:
    - name: pooling_modes
      params: {model.pooling.mode: [gap, attn, hybrid]}
    - name: alpha_sensitivity
      params: {eval.scoring.alpha: [0.0, 0.3, 0.5, 0.7, 1.0]}
```

## Cross-Dataset Generalization

Test generalization by training on one dataset and testing on others:

```yaml
cross_dataset:
  enabled: true
  train_on: shanghaitech
  test_on: [ucsd_ped2, avenue]
```

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` if OOM errors occur
2. **Speed**: Use `model.yolo.weights=yolov8n.pt` (nano) for faster extraction
3. **Accuracy**: Use `model.yolo.weights=yolov8l.pt` (large) for better features
4. **Sequence Length**: Increase `data.sequence.T` for longer temporal context
5. **Pooling Mode**: `hybrid` generally performs best but is slower

## Baseline Comparisons (TAO Paper)

### UCSD Ped2
| Method | Frame-AUC | Pixel-AUC |
|--------|-----------|-----------|
| OCAD | 96.2 | 97.8 |
| BAF-AT | 96.5 | 98.1 |
| AED-SSMTL | 97.2 | 98.5 |
| HF2VAD | 99.3 | 99.1 |
| STPT | 96.9 | 98.4 |

### ShanghaiTech
| Method | Frame-AUC | Pixel-AUC |
|--------|-----------|-----------|
| OCAD | 84.8 | 89.2 |
| BAF-AT | 92.3 | 93.7 |
| AED-SSMTL | 97.4 | 97.8 |
| HF2VAD | 91.1 | 93.5 |
| STPT | 96.5 | 97.2 |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{tao2025,
  title={Track Any Anomalous Object},
  booktitle={CVPR},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Issue: "Cannot find YOLOv8 weights"
**Solution**: Download weights first:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or sequence length:
```bash
python train.py train.batch_size=32 data.sequence.T=12
```

### Issue: "No features extracted"
**Solution**: Check video paths and formats in data config. Ensure videos are readable by OpenCV.

### Issue: "Wandb login required"
**Solution**: Login to wandb or disable it:
```bash
wandb login
# or
python train.py wandb.mode=disabled
```

## Contributing

Contributions welcome! Please open an issue or pull request.

## Acknowledgments

- YOLOv8 by Ultralytics
- ByteTrack implementation in Ultralytics
- TAO paper authors for evaluation protocol
- PyTorch team for the framework

## Contact

For questions or issues, please open a GitHub issue.
