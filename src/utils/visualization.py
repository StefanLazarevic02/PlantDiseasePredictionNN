import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from pathlib import Path
from typing import List
import torch


def show_augmentations(
    image_path: str,
    transform: A.Compose,
    n_examples: int = 8,
    figsize: tuple = (16, 8),
) -> None:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    n_cols = 4
    n_rows = (n_examples + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    axes[0].imshow(image)
    axes[0].set_title("ORIGINAL", fontsize=10, fontweight="bold", color="blue")
    axes[0].axis("off")

    vis_transform = A.Compose([
        t for t in transform.transforms
        if not isinstance(t, (A.Normalize, A.pytorch.transforms.ToTensorV2))
    ])

    for i in range(1, n_examples + 1):
        augmented = vis_transform(image=image)["image"]
        axes[i].imshow(augmented)
        axes[i].set_title(f"Aug #{i}", fontsize=9)
        axes[i].axis("off")

    for i in range(n_examples + 1, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Augmentation: {Path(image_path).parent.name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_batch(dataloader, class_names: List[str], n_images: int = 16) -> None:
    images, labels = next(iter(dataloader))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    n_images = min(n_images, images.size(0))
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i in range(n_images):
        img = images[i] * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        label_idx = labels[i].item()
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
        short_name = class_name.replace("___", "\n").replace("__", "\n")

        axes[i].imshow(img)
        axes[i].set_title(short_name, fontsize=7)
        axes[i].axis("off")

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Batch Preview (denormalized)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()