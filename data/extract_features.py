#!/usr/bin/env python3
import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

from data_prep import load_splits, build_transforms

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)


def get_device(preferred="cuda"):
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device("cpu")


def build_eval_dataloaders(data_root, splits_dir, batch_size=64, num_workers=4):
    splits_dir = Path(splits_dir)
    train_idx, val_idx, test_idx = load_splits(splits_dir)

    # Use eval transform for deterministic features
    _, eval_tf = build_transforms()

    full_dataset = datasets.ImageFolder(
        root=str(data_root),
        transform=eval_tf,
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    class_names = full_dataset.classes
    return train_loader, val_loader, test_loader, class_names


def build_resnet_feature_extractor():
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        print("Loaded ResNet18 with ResNet18_Weights.IMAGENET1K_V1")
    except (ImportError, AttributeError):
        model = models.resnet18(pretrained=True)
        print("Loaded ResNet18 with pretrained=True (old torchvision API)")

    backbone = nn.Sequential(*list(model.children())[:-1])  # [B, 512, 1, 1]
    backbone.eval()
    return backbone


@torch.no_grad()
def extract_split_features(loader, backbone, device):
    features_list = []
    labels_list = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats = backbone(images)            # [B, 512, 1, 1]
        feats = feats.squeeze(-1).squeeze(-1)  # [B, 512]

        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Extract CNN features for ML models.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to root of images (folder-per-class).")
    parser.add_argument("--splits_dir", type=str, required=True,
                        help="Directory containing split .npy files and class_weights.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save .npz feature files.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu' (default: cuda, falls back to cpu if necessary).")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Building dataloaders")
    train_loader, val_loader, test_loader, class_names = build_eval_dataloaders(
        data_root=data_root,
        splits_dir=splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Building ResNet18 feature extractor")
    backbone = build_resnet_feature_extractor().to(device)

    print("Extracting train features")
    X_train, y_train = extract_split_features(train_loader, backbone, device)
    print(f"Train features shape: {X_train.shape}")

    print("Extracting val features...")
    X_val, y_val = extract_split_features(val_loader, backbone, device)
    print(f"Val features shape: {X_val.shape}")

    print("Extracting test features...")
    X_test, y_test = extract_split_features(test_loader, backbone, device)
    print(f"Test features shape: {X_test.shape}")

    np.savez(out_dir / "train_features.npz", X=X_train, y=y_train)
    np.savez(out_dir / "val_features.npz", X=X_val, y=y_val)
    np.savez(out_dir / "test_features.npz", X=X_test, y=y_test)

    # Save class names for convenience
    with open(out_dir / "class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"Saved features to {out_dir}")


if __name__ == "__main__":
    main()
