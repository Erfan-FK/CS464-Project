#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn

# You may need to: pip install scikit-learn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Make project root importable (same pattern as in train script)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from data.data_prep import get_dataloaders  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda"):
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device("cpu")


class CNN(nn.Module):
    """
    Same architecture as your training script.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # 3 x 224 x 224 -> 64 x 112 x 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64 x 112 x 112 -> 128 x 56 x 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 128 x 56 x 56 -> 256 x 28 x 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),  # -> 256 x 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 256*4*4 = 4096
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_checkpoint(path: Path, device: torch.device):
    """
    Same semantics as in your training script; tries weights_only=True first,
    falls back if needed.
    """
    try:
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            # Older PyTorch without weights_only
            return torch.load(path, map_location=device)
        except Exception:
            print("Warning: weights_only=True failed, retrying with weights_only=False")
            return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint from {path}: {e}")
        return None


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN on the test set with detailed metrics."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to root of images (folder-per-class), same as in training.",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Directory with train/val/test indices (e.g., contains test_indices.npy).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where checkpoints were saved during training.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional: explicit path to checkpoint. "
             "If not given, will use OUT_DIR/cnn_best.pt, then OUT_DIR/cnn_last.pt.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    data_root = args.data_root
    splits_dir = args.splits_dir
    out_dir = Path(args.out_dir)

    # Build dataloaders just like in training to get the exact same test split
    print("Building dataloaders (only test loader will be used)...")
    _, _, test_loader, class_weights, class_names = get_dataloaders(
        data_root=data_root,
        splits_dir=splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=False,  # no weighting needed for test
    )

    num_classes = len(class_names)
    print(f"#classes: {num_classes}")

    # Build model and load checkpoint
    model = CNN(num_classes=num_classes).to(device)

    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
    else:
        ckpt_path = out_dir / "cnn_best.pt"

    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found. Trying cnn_last.pt instead...")
        alt = out_dir / "cnn_last.pt"
        if alt.exists():
            ckpt_path = alt
        else:
            print(f"Neither cnn_best.pt nor cnn_last.pt found in {out_dir}. Exiting.")
            sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, device)
    if ckpt is None:
        print("Failed to load checkpoint; exiting.")
        sys.exit(1)

    model.load_state_dict(ckpt["model_state_dict"])

    # Collect predictions on test set
    print("Running inference on test set...")
    y_true, y_pred = collect_predictions(model, test_loader, device)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n=== Test set metrics ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Micro F1     : {micro_f1:.4f}")
    print(f"Weighted F1  : {weighted_f1:.4f}")

    print("\n=== Per-class metrics (good for imbalance) ===")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("Class order:", class_names)


if __name__ == "__main__":
    main()
