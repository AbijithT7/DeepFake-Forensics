# Notebooks Directory

This directory contains Jupyter notebooks used for exploratory data analysis (EDA), rapid experimentation, data visualization, and model prototyping for **DeepFake Forensics**.

## Structure & Naming Conventions

```text
notebooks/
└── README.md     # Guidelines for exploratory notebooks
```

Recommended notebook prefixing:
- `01_eda_*.ipynb`: Initial exploratory data analysis on real vs. fake datasets.
- `02_preprocessing_*.ipynb`: Face extraction, alignment, and data transformation pipelines.
- `03_baseline_model_*.ipynb`: Baseline classifier experiments and benchmark metrics.
- `04_evaluation_*.ipynb`: Model evaluation, error analysis, and confusion matrices.
- `05_explainability_*.ipynb`: Grad-CAM, attention map, and saliency visualization experiments.

## Guidelines

- Clear outputs before committing if file sizes are large.
- Automatic notebook checkpoints (`.ipynb_checkpoints/`) are ignored by `.gitignore`.
- Production-ready logic should be refactored into the `src/` modules.
