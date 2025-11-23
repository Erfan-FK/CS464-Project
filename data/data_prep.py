#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

IMG_SIZE = 224
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 42


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return train_transform, eval_transform


def create_splits(data_root, splits_dir):
    data_root = str(data_root)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.ImageFolder(
        root=data_root,
        transform=transforms.ToTensor()
    )

    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    class_names = dataset.classes
    num_classes = len(class_names)

    assert 0 < TRAIN_RATIO < 1
    assert 0 < VAL_RATIO < 1
    test_ratio = 1.0 - TRAIN_RATIO - VAL_RATIO
    if test_ratio <= 0:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be < 1.")

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        targets,
        test_size=(VAL_RATIO + test_ratio),
        stratify=targets,
        random_state=SEED,
    )

    val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + test_ratio)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=y_temp,
        random_state=SEED,
    )

    np.save(splits_dir / "train_indices.npy", train_idx)
    np.save(splits_dir / "val_indices.npy", val_idx)
    np.save(splits_dir / "test_indices.npy", test_idx)

    train_labels = targets[train_idx]
    val_labels = targets[val_idx]
    test_labels = targets[test_idx]

    total_counts = Counter(targets.tolist())
    train_counts = Counter(train_labels.tolist())
    val_counts = Counter(val_labels.tolist())
    test_counts = Counter(test_labels.tolist())

    total_train = len(train_labels)

    class_weights = []
    for c in range(num_classes):
        count_c = train_counts.get(c, 1)
        class_weights.append(total_train / (num_classes * count_c))

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    torch.save(class_weights, splits_dir / "class_weights.pt")

    class_stats = {}
    for idx, name in enumerate(class_names):
        class_stats[name] = {
            "index": int(idx),
            "total": int(total_counts.get(idx, 0)),
            "train": int(train_counts.get(idx, 0)),
            "val": int(val_counts.get(idx, 0)),
            "test": int(test_counts.get(idx, 0)),
            "weight": float(class_weights[idx].item()),
        }

    with open(splits_dir / "class_counts.json", "w") as f:
        json.dump(
            {
                "class_names": class_names,
                "classes": class_stats,
                "total_samples": int(len(targets)),
                "total_train_samples": int(total_train),
                "total_val_samples": int(len(val_labels)),
                "total_test_samples": int(len(test_labels)),
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "test_ratio": test_ratio,
                "seed": SEED,
            },
            f,
            indent=2,
        )

    print(f"Created splits in {splits_dir}")
    print(f"#classes: {num_classes}")
    print(f"Total samples: {len(targets)}")
    print(f"Train/Val/Test sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")
    print("Per-class stats (index, name, total, train, val, test, weight):")

    for idx, name in sorted(enumerate(class_names), key=lambda x: x[1].lower()):
        stats = class_stats[name]
        print(
            f"  [{stats['index']:2d}] {name:30s} "
            f"total={stats['total']:4d} "
            f"train={stats['train']:4d} "
            f"val={stats['val']:4d} "
            f"test={stats['test']:4d} "
            f"weight={stats['weight']:.4f}"
        )

    return train_idx, val_idx, test_idx, train_labels.tolist(), class_names


def load_splits(splits_dir):
    splits_dir = Path(splits_dir)
    train_idx = np.load(splits_dir / "train_indices.npy")
    val_idx = np.load(splits_dir / "val_indices.npy")
    test_idx = np.load(splits_dir / "test_indices.npy")
    return train_idx, val_idx, test_idx


def load_class_weights(splits_dir):
    splits_dir = Path(splits_dir)
    class_weights = torch.load(splits_dir / "class_weights.pt", map_location="cpu")
    return class_weights


def get_dataloaders(
    data_root,
    splits_dir,
    batch_size=32,
    num_workers=4,
    use_weighted_sampler=True,
):
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not (splits_dir / "train_indices.npy").exists():
        print("Splits not found, creating them...")
        train_idx, val_idx, test_idx, train_labels_list, class_names = create_splits(
            data_root=data_root,
            splits_dir=splits_dir,
        )
    else:
        print("Loading existing splits...")
        train_idx, val_idx, test_idx = load_splits(splits_dir)
        dataset = datasets.ImageFolder(
            root=data_root,
            transform=transforms.ToTensor()
        )
        class_names = dataset.classes

    class_weights = load_class_weights(splits_dir)

    train_tf, eval_tf = build_transforms()

    full_train_dataset = datasets.ImageFolder(
        root=data_root,
        transform=train_tf,
    )
    full_eval_dataset = datasets.ImageFolder(
        root=data_root,
        transform=eval_tf,
    )

    from numpy import array as _array  # to satisfy lints if needed

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_eval_dataset, val_idx)
    test_dataset = Subset(full_eval_dataset, test_idx)

    full_targets = np.array(full_train_dataset.targets)
    train_targets = full_targets[train_idx]

    if use_weighted_sampler:
        weights_np = class_weights.numpy()
        sample_weights = torch.tensor(
            [weights_np[label] for label in train_targets],
            dtype=torch.float32,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("DataLoaders ready.")
    print(f"#train batches: {len(train_loader)}, "
          f"#val batches: {len(val_loader)}, #test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights, class_names


def main():
    parser = argparse.ArgumentParser(description="Prepare data splits and class weights.")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to root of house-plant-species images (folder-per-class).",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="./splits",
        help="Directory to store split indices and class weights.",
    )
    args = parser.parse_args()

    create_splits(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
