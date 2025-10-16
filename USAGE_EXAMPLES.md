# Usage Examples

This document provides detailed examples for common use cases.

## Table of Contents
1. [Basic Workflow](#basic-workflow)
2. [Advanced Configuration](#advanced-configuration)
3. [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Custom Datasets](#custom-datasets)
5. [Debugging and Visualization](#debugging-and-visualization)

---

## Basic Workflow

### Complete Pipeline from Scratch

```bash
# 1. Extract features from ShanghaiTech dataset
python extract/offline_extract.py \
    data.datasets=[shanghaitech] \
    data.root=data/shanghaitech \
    model.yolo.weights=yolov8s.pt \
    model.pooling.mode=hybrid

# 2. Train on normal videos
python train.py \
    data.datasets=[shanghaitech] \
    train.epochs=50 \
    wandb.name=shanghaitech_baseline

# 3. Evaluate on test set
python eval.py \
    data.datasets=[shanghaitech] \
    eval.checkpoint=best

# 4. Run inference on new video
python infer.py \
    source=data/shanghaitech/testing/frames/video_01.avi \
    viz.enable=true
```

---

## Advanced Configuration

### Multi-Dataset Training

Train on multiple datasets simultaneously:

```bash
python train.py \
    data.datasets=[shanghaitech,ucsd_ped2,avenue] \
    train.epochs=100 \
    train.batch_size=128 \
    wandb.name=multi_dataset_training
```

### Custom Pooling Configuration

#### GAP (Global Average Pooling)
```bash
python train.py \
    model.pooling.mode=gap \
    wandb.name=gap_pooling
```

#### Attention Pooling
```bash
python train.py \
    model.pooling.mode=attn \
    model.pooling.attn_heads=4 \
    model.pooling.attn_dropout=0.1 \
    wandb.name=attention_pooling
```

#### Hybrid Pooling (Recommended)
```bash
python train.py \
    model.pooling.mode=hybrid \
    model.pooling.attn_heads=1 \
    wandb.name=hybrid_pooling
```

### Temporal Configuration

Vary sequence length and stride:

```bash
# Longer temporal context
python train.py \
    data.sequence.T=32 \
    data.sequence.stride=2 \
    wandb.name=long_sequences

# Shorter sequences (faster)
python train.py \
    data.sequence.T=8 \
    data.sequence.stride=1 \
    wandb.name=short_sequences
```

### Model Architecture Variants

```bash
# Deeper model
python train.py \
    model.temporal.depth=12 \
    model.temporal.num_heads=16 \
    wandb.name=deep_model

# Wider model
python train.py \
    model.temporal.d_model=512 \
    model.pooling.d_model=512 \
    wandb.name=wide_model

# Smaller, faster model
python train.py \
    model.temporal.depth=4 \
    model.temporal.num_heads=4 \
    model.temporal.d_model=128 \
    model.pooling.d_model=128 \
    wandb.name=small_model
```

---

## Hyperparameter Tuning

### Learning Rate Scheduling

```bash
# Cosine annealing (default)
python train.py \
    train.scheduler.name=cosine \
    train.scheduler.warmup_epochs=5 \
    train.scheduler.min_lr=1e-6

# Step decay
python train.py \
    train.scheduler.name=step \
    train.scheduler.step_size=20 \
    train.scheduler.gamma=0.5

# No scheduler
python train.py \
    train.scheduler.name=none
```

### Loss Weights

```bash
# MAE-focused
python train.py \
    train.losses.lambda_mae=1.0 \
    train.losses.lambda_cl=0.1 \
    wandb.name=mae_focused

# Contrastive-focused
python train.py \
    train.losses.lambda_mae=0.5 \
    train.losses.lambda_cl=0.5 \
    wandb.name=contrastive_focused

# With forecasting
python train.py \
    model.heads.use_forecasting=true \
    model.heads.forecasting_steps=4 \
    train.losses.lambda_forecast=0.1 \
    wandb.name=with_forecasting
```

### Masking Strategy

```bash
# High masking ratio (more challenging)
python train.py \
    model.temporal.mask_ratio=0.90 \
    wandb.name=high_masking

# Low masking ratio
python train.py \
    model.temporal.mask_ratio=0.50 \
    wandb.name=low_masking
```

---

## Custom Datasets

### Adding a New Dataset

1. Update `configs/data.yaml`:

```yaml
my_dataset:
  root: ${data.root}/my_dataset
  train_split: train
  test_split: test
  annotations: annotations
  video_format: "*.mp4"
  frame_format: "*.jpg"
  threshold: 1.5
```

2. Extract features:

```bash
python extract/offline_extract.py \
    data.datasets=[my_dataset] \
    data.my_dataset.root=data/my_dataset
```

3. Train and evaluate:

```bash
python train.py data.datasets=[my_dataset]
python eval.py data.datasets=[my_dataset]
```

### Frame Sequences as Videos

If your videos are stored as image sequences:

```bash
# Organize as:
# data/my_dataset/train/video_01/frame_0001.jpg
# data/my_dataset/train/video_01/frame_0002.jpg
# ...

python extract/offline_extract.py \
    data.datasets=[my_dataset] \
    data.my_dataset.video_format="*/" \
    data.my_dataset.frame_format="*.jpg"
```

---

## Debugging and Visualization

### Training with Frequent Logging

```bash
python train.py \
    train.logging.log_every_n_steps=5 \
    train.logging.log_samples=true \
    train.logging.num_samples=16 \
    train.checkpoint.save_every=1 \
    wandb.mode=online
```

### Offline Mode (No Internet)

```bash
python train.py wandb.mode=offline
# Later, sync to wandb
wandb sync logs/
```

### Disable Wandb Completely

```bash
python train.py wandb.mode=disabled
```

### Evaluation with Detailed Outputs

```bash
python eval.py \
    eval.output.save_predictions=true \
    eval.output.save_per_frame_csv=true \
    eval.output.save_per_object_csv=true \
    eval.output.save_html_report=true
```

### Visualizations Only

```bash
# Create visualizations from existing results
python -c "
from viz.visualizer import AnomalyVisualizer
import numpy as np

viz = AnomalyVisualizer('results/visualizations')

# Load your results
frame_scores = np.load('results/frame_scores.npy')

# Generate plots
viz.plot_frame_scores(frame_scores, threshold=0.5)
"
```

### Ablation Study

```bash
python eval.py \
    eval.ablation.enabled=true \
    eval.ablation.studies[0].name=pooling_modes \
    eval.ablation.studies[0].params={model.pooling.mode:[gap,attn,hybrid]}
```

---

## Performance Optimization

### Fast Extraction (Nano YOLO)

```bash
python extract/offline_extract.py \
    model.yolo.weights=yolov8n.pt \
    model.pooling.mode=gap
```

### High Accuracy (Large YOLO)

```bash
python extract/offline_extract.py \
    model.yolo.weights=yolov8l.pt \
    model.pooling.mode=hybrid
```

### Mixed Precision Training

```bash
python train.py \
    train.mixed_precision.enabled=true \
    train.mixed_precision.dtype=float16
```

### Gradient Accumulation (Large Effective Batch)

```bash
python train.py \
    train.batch_size=32 \
    train.accumulation_steps=4  # Effective batch size = 128
```

---

## Evaluation Scenarios

### Threshold Sensitivity Analysis

```bash
# EVT thresholding
python eval.py eval.threshold.method=evt

# Fixed thresholds
python eval.py eval.threshold.method=fixed eval.threshold.fixed_value=1.5

# Percentile thresholding
python eval.py eval.threshold.method=percentile eval.threshold.percentile=95
```

### Cross-Dataset Evaluation

```bash
# Train on ShanghaiTech, test on UCSD and Avenue
python train.py data.datasets=[shanghaitech]
python eval.py data.datasets=[ucsd_ped2,avenue]
```

### Compare with Baselines

Evaluation automatically compares with TAO paper baselines. Results printed in table format.

---

## Production Deployment

### Export Model

```python
import torch
from models import TemporalMAE

# Load trained model
model = TemporalMAE(d_model=256, depth=6, num_heads=8)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export to TorchScript
scripted = torch.jit.script(model)
scripted.save('temporal_mae_scripted.pt')
```

### Batch Inference

```bash
# Process multiple videos
for video in data/test_videos/*.mp4; do
    python infer.py source="$video" viz.enable=true
done
```

### Real-Time Processing

```bash
python infer.py \
    source=0 \  # Webcam
    model.yolo.weights=yolov8n.pt \
    data.sequence.T=8 \
    viz.enable=true
```

---

## Common Issues and Solutions

### Issue: Low Frame-AUC

**Try:**
- Increase sequence length: `data.sequence.T=24`
- Use hybrid pooling: `model.pooling.mode=hybrid`
- Train longer: `train.epochs=100`
- Increase contrastive weight: `train.losses.lambda_cl=0.5`

### Issue: High False Positives

**Try:**
- Adjust threshold: `eval.threshold.fixed_value=2.0`
- Use EVT thresholding: `eval.threshold.method=evt`
- Increase temporal context: `data.sequence.T=32`

### Issue: Slow Inference

**Try:**
- Use smaller YOLO: `model.yolo.weights=yolov8n.pt`
- Reduce sequence length: `data.sequence.T=8`
- Use GAP pooling: `model.pooling.mode=gap`
- Decrease stride: `data.sequence.stride=2`

---

## Advanced Topics

### Custom Loss Functions

Modify `models/losses.py` to add custom losses, then:

```bash
python train.py \
    train.losses.lambda_custom=0.2
```

### Multi-GPU Training

```bash
python train.py \
    train.distributed.enabled=true \
    train.distributed.backend=nccl
```

### Resume Training

```bash
python train.py \
    resume_from=checkpoints/checkpoint_epoch_30.pt
```

---

For more details, see the main [README.md](README.md).
