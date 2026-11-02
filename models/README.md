# Models Directory

This directory stores deep learning model architectures, trained checkpoints, and exported inference weights for **DeepFake Forensics**.

## Structure & Conventions

```text
models/
└── README.md     # Model registry documentation and storage guidelines
```

## Checkpoint Management

- **Model Checkpoints**: Weights (`.pt`, `.pth`, `.onnx`, `.safetensors`, `.ckpt`) are excluded from Git via `.gitignore`.
- **Planned Architectures**:
  - EfficientNet-based backbones (e.g., EfficientNet-B4/B7) for spatial artifact detection.
  - Recurrent/temporal networks (e.g., LSTM, GRU, Video Transformers) for frame-to-frame temporal inconsistency analysis.
  - Frequency-domain feature extraction heads (e.g., FFT / DCT analysis).
- **Naming Convention (Planned)**:
  - `<architecture>_<dataset>_<metric>_<timestamp>.pt`
  - Example: `efficientnet_b4_faceforensics_val_acc94_20261103.pt`
