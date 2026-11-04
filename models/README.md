# Models Directory

This directory stores deep learning model weights and detectors for **DeepFake Forensics**.

## Production Model Checkpoints

1. **`efficientnet_b0_best.pth` (Primary Deepfake Classifier)**
   - Architecture: Pretrained EfficientNet-B0 with Global Average Pooling, Dropout, and binary classification head.
   - Input: $224 \times 224 \times 3$, ImageNet normalization (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
   - Dataset: RVF10K (8020 split: 8,000 train, 1,000 val, 1,000 test).
   - Test Metrics: 99.30% Accuracy, 99.99% ROC-AUC, 0.993 Weighted F1.
   - Explainability: Terminal convolutional layer `backbone.features[8]` generates standard Grad-CAM attributions.
   - Integrity: Original trained weights preserved untouched.

2. **`scrfd_500m.onnx` (SCRFD Face Detector)**
   - Architecture: Sample and Computation Redistribution for Efficient Face Detection (SCRFD 500M).
   - Runtime: ONNX Runtime CPU / CUDA execution.
   - Function: High-accuracy facial localization with 5 facial keypoints. Used to compute isolated face crops with 20% contextual margin for classification.
