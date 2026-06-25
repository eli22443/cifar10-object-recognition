"""CIFAR-10 CNN classifier utilities for ADL HW4 Task 1."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.sgd import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

DEFAULT_DATA_ROOT = "./data"
DEFAULT_TRAIN_SUBSET = 6000
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 40
DEFAULT_LR = 0.001
DEFAULT_MOMENTUM = 0.9


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Net(nn.Module):
    """CNN from the PyTorch CIFAR-10 tutorial / Lecture 7."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


@dataclass
class Task1Results:
    train_subset_size: int
    epochs: int
    final_train_acc: float
    final_test_acc: float
    per_class_acc: Dict[str, float]
    history: List[EpochMetrics]


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return images * 0.5 + 0.5


def make_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    train_subset_size: Optional[int] = DEFAULT_TRAIN_SUBSET,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    transform = get_transform()

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform
    )

    if train_subset_size is not None and train_subset_size < len(trainset):
        rng = random.Random(seed)
        indices = rng.sample(range(len(trainset)), train_subset_size)
        trainset = Subset(trainset, indices)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    momentum: float = DEFAULT_MOMENTUM,
) -> List[EpochMetrics]:
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    history: List[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, trainloader, criterion, device, optimizer=optimizer
        )
        test_loss, test_acc = run_epoch(model, testloader, criterion, device)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )
        history.append(metrics)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {train_loss:.3f} acc {100 * train_acc:.1f}% | "
                f"test loss {test_loss:.3f} acc {100 * test_acc:.1f}%"
            )

    return history


def per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    correct_pred = {name: 0 for name in CIFAR10_CLASSES}
    total_pred = {name: 0 for name in CIFAR10_CLASSES}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions.cpu()):
                classname = CIFAR10_CLASSES[label]
                total_pred[classname] += 1
                if label == prediction:
                    correct_pred[classname] += 1

    return {
        classname: 100.0 * correct_pred[classname] / total_pred[classname]
        for classname in CIFAR10_CLASSES
    }


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_images: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    images_out: List[torch.Tensor] = []
    labels_out: List[torch.Tensor] = []
    preds_out: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            images_out.append(images.cpu())
            labels_out.append(labels)
            preds_out.append(preds.cpu())
            if sum(batch.size(0) for batch in images_out) >= max_images:
                break

    images_cat = torch.cat(images_out)[:max_images]
    labels_cat = torch.cat(labels_out)[:max_images]
    preds_cat = torch.cat(preds_out)[:max_images]
    return images_cat, labels_cat, preds_cat


def save_results(results: Task1Results, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(results), f, indent=2)


def plot_accuracy_curves(history: Sequence[EpochMetrics], path: str) -> None:
    epochs = [m.epoch for m in history]
    train_acc = [100 * m.train_acc for m in history]
    test_acc = [100 * m.test_acc for m in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    figure = cast(Figure, fig)
    axis = cast(Axes, ax)
    axis.plot(epochs, train_acc, marker="o", label="Train")
    axis.plot(epochs, test_acc, marker="o", label="Test")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("CIFAR-10 classification accuracy")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_sample_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    path: str,
) -> None:
    n = images.size(0)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes_raw = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    figure = cast(Figure, fig)
    axes = cast(List[Axes], list(np.atleast_1d(axes_raw).ravel()))

    for idx in range(n):
        img = denormalize(images[idx]).numpy().transpose(1, 2, 0)
        true_name = CIFAR10_CLASSES[labels[idx]]
        pred_name = CIFAR10_CLASSES[predictions[idx]]
        color = "green" if labels[idx] == predictions[idx] else "red"
        axes[idx].imshow(np.clip(img, 0, 1))
        axes[idx].set_title(f"true: {true_name}\npred: {pred_name}", color=color, fontsize=8)
        axes[idx].axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    figure.suptitle("Test images with predicted labels", fontsize=11)
    figure.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_per_class_table(per_class: Dict[str, float], path: str) -> None:
    names = list(CIFAR10_CLASSES)
    values = [per_class[name] for name in names]

    fig, ax = plt.subplots(figsize=(5, 4))
    figure = cast(Figure, fig)
    axis = cast(Axes, ax)
    axis.axis("off")
    table = axis.table(
        cellText=[[name, f"{acc:.1f}%"] for name, acc in zip(names, values)],
        colLabels=["Class", "Accuracy"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    axis.set_title("Per-class test accuracy", pad=20)
    figure.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
