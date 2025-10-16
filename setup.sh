#!/bin/bash

# Setup script for Temporal MAE project

echo "=========================================="
echo "Temporal MAE Setup Script"
echo "=========================================="

# Create conda environment
echo "Creating conda environment..."
conda create -n temporal_mae python=3.9 -y
source activate temporal_mae

# Install PyTorch (adjust for your CUDA version)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "Creating directory structure..."
mkdir -p data
mkdir -p features
mkdir -p checkpoints
mkdir -p results/videos
mkdir -p results/visualizations
mkdir -p results/csv
mkdir -p results/reports
mkdir -p logs

# Download YOLOv8 weights
echo "Downloading YOLOv8 weights..."
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -P ./

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place your datasets in the 'data/' directory"
echo "2. Run feature extraction: python extract/offline_extract.py"
echo "3. Train model: python train.py"
echo "4. Evaluate: python eval.py"
echo "5. Inference: python infer.py source=your_video.mp4"
echo ""
echo "For more information, see README.md"
echo "=========================================="
