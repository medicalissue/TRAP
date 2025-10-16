# Quick Reference Guide

## Common Commands

### Feature Extraction
```bash
# Basic extraction
python extract/offline_extract.py data.datasets=[shanghaitech]

# Custom YOLO model
python extract/offline_extract.py model.yolo.weights=yolov8l.pt

# Different pooling
python extract/offline_extract.py model.pooling.mode=gap
```

### Training
```bash
# Basic training
python train.py data.datasets=[shanghaitech]

# Multi-dataset
python train.py data.datasets=[shanghaitech,ucsd_ped2,avenue]

# Adjust hyperparameters
python train.py train.epochs=100 train.batch_size=128 train.optimizer.lr=5e-5
```

### Evaluation
```bash
# Evaluate best model
python eval.py data.datasets=[shanghaitech] eval.checkpoint=best

# Different threshold
python eval.py eval.threshold.method=fixed eval.threshold.fixed_value=2.0

# Save outputs
python eval.py eval.output.save_predictions=true
```

### Inference
```bash
# Process video
python infer.py source=video.mp4 viz.enable=true

# Webcam
python infer.py source=0
```

## Configuration Overrides

### Model Architecture
```bash
# Deeper model
model.temporal.depth=12 model.temporal.num_heads=16

# Wider model
model.temporal.d_model=512 model.pooling.d_model=512

# Longer sequences
data.sequence.T=32 data.sequence.stride=2
```

### Training Settings
```bash
# Learning rate
train.optimizer.lr=1e-4

# Loss weights
train.losses.lambda_mae=1.0 train.losses.lambda_cl=0.3

# Masking ratio
model.temporal.mask_ratio=0.75
```

### Evaluation Settings
```bash
# Scoring balance
eval.scoring.alpha=0.5

# Smoothing
eval.scoring.beta=0.9

# Aggregation
eval.scoring.aggregation=max  # or mean
```

## File Locations

### Inputs
- **Data**: `data/`
- **Configs**: `configs/`
- **Checkpoints**: `checkpoints/`

### Outputs
- **Features**: `features/`
- **Results**: `results/`
- **Logs**: `logs/`
- **Wandb**: `wandb/`

## Important Parameters

### Model
| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `model.pooling.mode` | `hybrid` | gap, attn, hybrid | Feature pooling |
| `model.temporal.depth` | 6 | 4-12 | Transformer layers |
| `model.temporal.num_heads` | 8 | 4-16 | Attention heads |
| `data.sequence.T` | 16 | 8-32 | Sequence length |

### Training
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `train.epochs` | 50 | 10-200 | Training epochs |
| `train.batch_size` | 64 | 16-256 | Batch size |
| `train.optimizer.lr` | 1e-4 | 1e-5 to 1e-3 | Learning rate |
| `train.losses.lambda_cl` | 0.3 | 0.0-1.0 | Contrastive weight |

### Evaluation
| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `eval.scoring.alpha` | 0.5 | 0.0-1.0 | Recon vs similarity |
| `eval.threshold.method` | evt | evt, fixed, percentile | Threshold method |

## Metrics Dictionary

### Pixel-Level
- **pixel_auroc**: Area under ROC curve (pixel-wise)
- **pixel_ap**: Average precision (pixel-wise)
- **pixel_aupro**: Area under per-region overlap
- **pixel_f1**: F1 score (pixel-wise)

### Object-Level
- **rbdc_f1**: Region-based detection F1
- **tbdc_f1**: Track-based detection F1

### Frame-Level
- **frame_auc**: Area under ROC (frame-wise)
- **frame_prauc**: Precision-Recall AUC (frame-wise)

## Troubleshooting

### Error: CUDA out of memory
```bash
# Reduce batch size
train.batch_size=32

# Reduce sequence length
data.sequence.T=8

# Use gradient accumulation
train.accumulation_steps=2
```

### Error: Cannot find checkpoint
```bash
# Check checkpoint directory
ls checkpoints/

# Specify full path
eval.checkpoint=checkpoints/best_model.pt
```

### Error: Dataset not found
```bash
# Check data root
data.root=path/to/data

# List available datasets
ls data/
```

## Hydra Features

### Override config groups
```bash
python train.py data=custom_data model=large_model
```

### Multi-run (sweeps)
```bash
python train.py -m train.optimizer.lr=1e-4,5e-5,1e-5
```

### Print config
```bash
python train.py --cfg job  # Print job config
python train.py --cfg all  # Print all configs
```

## Wandb Commands

### Login
```bash
wandb login
```

### Set project
```bash
python train.py wandb.project=my_project wandb.name=exp1
```

### Offline mode
```bash
python train.py wandb.mode=offline
# Later: wandb sync logs/
```

### Disable
```bash
python train.py wandb.mode=disabled
```

## Environment Variables

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set Python path
export PYTHONPATH=$PYTHONPATH:/path/to/temporal_mae

# Disable wandb
export WANDB_MODE=disabled
```

## Directory Structure Setup

```bash
mkdir -p data/{shanghaitech,ucsd_ped2,avenue}
mkdir -p features/{shanghaitech,ucsd_ped2,avenue}/{train,test}
mkdir -p checkpoints
mkdir -p results/{videos,visualizations,csv,reports}
mkdir -p logs
```

## Performance Optimization

### Fast extraction
```bash
model.yolo.weights=yolov8n.pt model.pooling.mode=gap
```

### Accurate extraction
```bash
model.yolo.weights=yolov8l.pt model.pooling.mode=hybrid
```

### Fast training
```bash
train.batch_size=128 data.sequence.T=8 model.temporal.depth=4
```

### Accurate training
```bash
train.epochs=100 data.sequence.T=24 model.temporal.depth=8
```

## Visualization Only

```python
from viz.visualizer import AnomalyVisualizer

viz = AnomalyVisualizer('results/viz')
viz.plot_frame_scores(frame_scores, threshold=0.5)
viz.create_anomaly_gallery(video_path, frame_scores, top_k=12)
```

## Export Model

```python
import torch
model.eval()
scripted = torch.jit.script(model)
scripted.save('model.pt')
```

## Useful Git Commands

```bash
# Clone
git clone https://github.com/yourusername/temporal_mae.git

# Ignore large files
echo "*.pt" >> .gitignore
echo "data/" >> .gitignore

# Track experiments
git add configs/ models/ train.py
git commit -m "Updated model architecture"
```
