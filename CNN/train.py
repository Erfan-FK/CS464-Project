#!/usr/bin/env python3
import argparse
import random
import sys
import csv
from pathlib import Path
import os

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from data.data_prep import get_dataloaders  # noqa: E402

import warnings
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred="cuda"):
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device("cpu")


class CNN(nn.Module):
    def __init__(self, num_classes):
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate_on_test(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc


def save_checkpoint(state, filename):
    try:
        torch.save(state, filename)
    except Exception as e:
        print(f"Error saving checkpoint to {filename}: {e}")


def load_checkpoint(path, device):
    try:
        # We use weights_only=True to avoid the FutureWarning and security risk.
        # However, this requires PyTorch 2.0+. If using older version, remove weights_only=True.
        # Given the user's error message, they are on a version that supports it (and warns about it).
        # If loading complex objects (like optimizer state sometimes), safe_globals might be needed.
        # But for standard training loops, weights_only=True usually works if just tensors/primitives.
        # IF IT FAILS, we fall back to False.
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
             # older pytorch doesn't have weights_only
             return torch.load(path, map_location=device)
        except Exception:
             # fallback if strict loading fails (though not ideal security-wise, it's what user had)
             print("Warning: weights_only=True failed, retrying with weights_only=False")
             return torch.load(path, map_location=device, weights_only=False)
             
    except Exception as e:
        print(f"Failed to load checkpoint from {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train a simple CNN from scratch on house plant dataset.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to root of images (folder-per-class).")
    parser.add_argument("--splits_dir", type=str, required=True,
                        help="Directory with train/val/test indices and class_weights.pt.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save model checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Total number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = args.data_root
    splits_dir = args.splits_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Building dataloaders")
    # get_dataloaders internally uses weights_only=True for class_weights now (after fix in data_prep.py)
    train_loader, val_loader, test_loader, class_weights, class_names = get_dataloaders(
        data_root=data_root,
        splits_dir=splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=False,
    )

    num_classes = len(class_names)
    print(f"#classes: {num_classes}")

    print("Building CNN model")
    model = CNN(num_classes=num_classes).to(device)

    # Class-weighted cross entropy to handle imbalance
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Add a scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )

    best_ckpt_path = out_dir / "cnn_best.pt"
    last_ckpt_path = out_dir / "cnn_last.pt"
    results_csv_path = out_dir / "results.csv"

    best_val_acc = 0.0
    start_epoch = 1
    
    # Initialize results file if not exists
    if not results_csv_path.exists():
        with open(results_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # Try to resume from last checkpoint first (to continue exactly where left off)
    ckpt = None
    if last_ckpt_path.exists():
        print(f"Found last checkpoint at {last_ckpt_path}, attempting to resume.")
        ckpt = load_checkpoint(last_ckpt_path, device)
    
    # If last checkpoint failed or didn't exist, try best checkpoint
    if ckpt is None and best_ckpt_path.exists():
        print(f"Found best checkpoint at {best_ckpt_path}, attempting to resume.")
        ckpt = load_checkpoint(best_ckpt_path, device)

    if ckpt is not None:
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc", 0.0))
            
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                
            print(f"Resuming from epoch {start_epoch} (best_val_acc={best_val_acc:.4f})")
        except Exception as e:
            print(f"Error restoring state from checkpoint: {e}")
            print("Starting from scratch due to checkpoint error.")
            start_epoch = 1
            best_val_acc = 0.0
            # Reset model and optimizer if partial load messed things up
            model = CNN(num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    else:
        print("No valid checkpoint found, starting from scratch.")

    if start_epoch > args.epochs:
        print(f"start_epoch ({start_epoch}) > total epochs ({args.epochs}); no further training will be done.")
    else:
        # Train
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = eval_one_epoch(
                model, val_loader, criterion, device
            )
            
            # Step scheduler
            scheduler.step(val_acc)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
            print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")
            print(f"LR now: {current_lr:.6f}")

            # Log results
            with open(results_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr])

            # Save last checkpoint (every epoch)
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "class_names": class_names,
                },
                last_ckpt_path
            )

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc,
                        "class_names": class_names,
                    },
                    best_ckpt_path
                )
                print(f"New best model saved to {best_ckpt_path} (val_acc={val_acc:.4f})")

        print("\nTraining done.")

    print(f"Best val acc: {best_val_acc:.4f}")

    # Load best model for test evaluation
    # Try to load best checkpoint, if it exists
    if best_ckpt_path.exists():
        print("Loading best model for test evaluation...")
        ckpt = load_checkpoint(best_ckpt_path, device)
        if ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            print("Could not load best checkpoint, using current model state.")
    else:
        print("Best checkpoint not found, using current model state.")

    test_acc = evaluate_on_test(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
