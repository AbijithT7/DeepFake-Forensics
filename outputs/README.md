# Outputs Directory

This directory stores artifacts generated during experiments, model evaluation, and inference runs.

## Directory Structure & Usage

```text
outputs/
└── README.md     # Guidelines for generated outputs
```

## Types of Generated Outputs

- **Visualizations**: Grad-CAM heatmaps, suspicious bounding boxes, facial attention overlays.
- **Evaluation Artifacts**: ROC curves, PR curves, confusion matrices, and classification reports.
- **Forensic Reports**: Generated PDF/JSON/HTML forensic analysis reports for analyzed media.
- **Logs**: Training and inference run logs.

> **Note**: Content inside this directory (excluding this `README.md` and `.gitkeep`) is ignored by Git to avoid tracking transient output files.
