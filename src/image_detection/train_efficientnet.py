"""
Training and Evaluation Pipeline for EfficientNet-B0 on RVF10K (8020 Configuration).

Features:
- Reproducible random seeds
- Pretrained EfficientNet-B0 initialization (ImageNet-1K)
- Standard ImageNet transforms with training augmentations
- Strict deterministic preprocessing for validation & testing
- Model checkpoint saving based on best validation performance
- Untouched test set evaluation producing classification report, confusion matrix, and ROC-AUC
- Export of training curves (PNG) and structured metrics (JSON)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detection.efficientnet_model import (
    CLASS_NAMES,
    DeepfakeEfficientNetB0,
    get_eval_transform,
    get_train_transform,
)

DEFAULT_DATASET_PATH = Path(r"C:\Users\abijith\Downloads\Real vs Fake Faces\Real vs Fake Faces\8020")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using device: CUDA ({device_name})")
        return torch.device("cuda")
    print("[INFO] Using device: CPU")
    return torch.device("cpu")


def create_dataloaders(
    dataset_dir: Path,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create training, validation, and test dataloaders from the 8020 dataset directory.
    """
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"

    for p in [train_dir, valid_dir, test_dir]:
        if not p.is_dir():
            raise FileNotFoundError(f"Dataset partition directory not found: {p}")

    train_dataset = ImageFolder(root=train_dir, transform=get_train_transform())
    valid_dataset = ImageFolder(root=valid_dir, transform=get_eval_transform())
    test_dataset = ImageFolder(root=test_dir, transform=get_eval_transform())

    print(f"[INFO] Dataset loaded from: {dataset_dir}")
    print(f"[INFO] Train samples: {len(train_dataset)} | Classes: {train_dataset.class_to_idx}")
    print(f"[INFO] Valid samples: {len(valid_dataset)}")
    print(f"[INFO] Test samples:  {len(test_dataset)}")

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )

    return train_loader, valid_loader, test_loader, train_dataset.class_to_idx


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train the model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Validation",
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a dataloader.
    Returns: loss, accuracy, roc_auc, all_preds, all_probs, all_targets
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"[{desc}]", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            # Probability of class 1 (REAL)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    # Compute ROC-AUC (class 1: REAL)
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs) * 100.0
    except Exception:
        epoch_auc = 0.0

    return (
        epoch_loss,
        epoch_acc,
        epoch_auc,
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_targets),
    )


def save_training_plots(history: Dict, output_path: Path) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    epochs = [h["epoch"] for h in history["epochs"]]
    train_loss = [h["train_loss"] for h in history["epochs"]]
    val_loss = [h["val_loss"] for h in history["epochs"]]
    train_acc = [h["train_acc"] for h in history["epochs"]]
    val_acc = [h["val_acc"] for h in history["epochs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss subplot
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss")
    ax1.plot(epochs, val_loss, "r-s", label="Validation Loss")
    ax1.set_title("Cross-Entropy Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy subplot
    ax2.plot(epochs, train_acc, "b-o", label="Train Accuracy (%)")
    ax2.plot(epochs, val_acc, "r-s", label="Validation Accuracy (%)")
    ax2.set_title("Accuracy vs. Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved training curves to: {output_path}")


def save_confusion_matrix_plot(cm: np.ndarray, classes: list, output_path: Path) -> None:
    """Plot and save confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Test Confusion Matrix (RVF10K 8020)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on RVF10K (8020).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to RVF10K 8020 directory containing train/, valid/, test/.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training/eval.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save logs and plots.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save model checkpoints.",
    )
    args = parser.parse_args()

    # 1. Setup
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 2. DataLoaders
    train_loader, valid_loader, test_loader, class_to_idx = create_dataloaders(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,  # 0 for Windows multiprocessing safety
    )

    # 3. Model, Criterion, Optimizer, Scheduler
    print("[INFO] Instantiating DeepfakeEfficientNetB0 (pretrained=True)...")
    model = DeepfakeEfficientNetB0(num_classes=2, pretrained=True, dropout_p=0.3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # 4. Training Loop (Validation only, NEVER touching test set)
    print("\n" + "=" * 70)
    print(f" STARTING {args.epochs}-EPOCH EFFICIENTNET-B0 TRAINING TRIAL (RVF10K 8020)")
    print("=" * 70)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_epoch = -1
    best_checkpoint_path = args.checkpoint_dir / "efficientnet_b0_best.pth"

    history = {
        "model": "EfficientNet-B0",
        "dataset": "RVF10K 8020",
        "seed": args.seed,
        "epochs": [],
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        # Validate (strictly on valid_loader)
        val_loss, val_acc, val_auc, _, _, _ = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            desc=f"Epoch {epoch} [Valid]",
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% AUC: {val_auc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpoint based on validation loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "class_to_idx": class_to_idx,
                    "classes": CLASS_NAMES,
                    "architecture": "EfficientNet-B0",
                    "input_size": [224, 224],
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint to: {best_checkpoint_path}")

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
                "val_auc": round(val_auc, 2),
                "lr": current_lr,
                "epoch_time_seconds": round(epoch_time, 2),
                "is_best": is_best,
            }
        )

    total_training_time = time.time() - start_time
    history["total_training_time_seconds"] = round(total_training_time, 2)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = round(best_val_loss, 4)
    history["best_val_acc"] = round(best_val_acc, 2)
    history["best_val_auc"] = round(best_val_auc, 2)

    # Save training history JSON
    history_path = args.output_dir / "efficientnet_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Saved training log to: {history_path}")

    # Save training curves plot
    curves_path = args.output_dir / "efficientnet_training_curves.png"
    save_training_plots(history, curves_path)

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE — LOADING BEST CHECKPOINT FOR FINAL TEST EVALUATION")
    print(f" Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)")
    print("=" * 70)

    # 5. Final Evaluation on UNTOUCHED Test Set
    checkpoint_data = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    test_loss, test_acc, test_auc, test_preds, test_probs, test_targets = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        desc="Final Test Evaluation",
    )

    report_dict = classification_report(
        test_targets,
        test_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    report_text = classification_report(
        test_targets,
        test_preds,
        target_names=CLASS_NAMES,
    )
    cm = confusion_matrix(test_targets, test_preds)

    # Save confusion matrix plot
    cm_path = args.output_dir / "efficientnet_confusion_matrix.png"
    save_confusion_matrix_plot(cm, CLASS_NAMES, cm_path)

    # Save test metrics JSON
    test_results = {
        "model": "EfficientNet-B0",
        "best_checkpoint": str(best_checkpoint_path),
        "evaluated_at_epoch": best_epoch,
        "test_dataset": "RVF10K 8020 (Untouched Test Split)",
        "test_samples": len(test_targets),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 2),
        "test_roc_auc": round(test_auc, 2),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    test_results_path = args.output_dir / "efficientnet_test_metrics.json"
    with open(test_results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"[INFO] Saved test evaluation metrics to: {test_results_path}")

    print("\n" + "=" * 70)
    print(" FINAL UNTOUCHED TEST SET METRICS (RVF10K 8020)")
    print("=" * 70)
    print(f" Test Accuracy:  {test_acc:.2f}%")
    print(f" Test ROC-AUC:   {test_auc:.2f}%\n")
    print(" Classification Report:")
    print(report_text)
    print(" Confusion Matrix:")
    print(f"  [TN (FAKE correctly identified): {cm[0, 0]} | FP (FAKE misclassified as REAL): {cm[0, 1]}]")
    print(f"  [FN (REAL misclassified as FAKE): {cm[1, 0]} | TP (REAL correctly identified): {cm[1, 1]}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
