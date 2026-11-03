# Models Directory

This directory stores deep learning model architectures, trained checkpoints, and exported inference weights for **DeepFake Forensics**.

## Available Checkpoints

1. **`deepfake_cnn.pth` (Model 1 - Custom CNN)**
   - Architecture: 4-layer Custom Convolutional Neural Network
   - Input: $128 \times 128 \times 3$, normalized with mean=0.5, std=0.5
   - Origin: Trained on Kaggle Deepfake vs Real 60K dataset

2. **`efficientnet_b0_best.pth` (Model 2 - EfficientNet-B0)**
   - Architecture: Pretrained EfficientNet-B0 with Global Average Pooling and Dropout head
   - Input: $224 \times 224 \times 3$, ImageNet normalization
   - Dataset: RVF10K (8020 configuration: 8,000 train, 1,000 val, 1,000 test)
   - Performance: 99.10% Val Accuracy, 99.30% Test Accuracy, 99.99% ROC-AUC

## Planned Architectures
- Spatial Artifact Analysis & Frequency-domain feature extraction
- Recurrent/temporal networks (e.g., LSTM, GRU, Video Transformers) for frame-to-frame temporal inconsistency analysis
- Multi-model calibrated ensemble blending Custom CNN and EfficientNet-B0 predictions

