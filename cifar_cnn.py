"""CIFAR-10 CNN classifier utilities for ADL HW4 Task 1."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.sgd import SGD
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


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return images * 0.5 + 0.5


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


def make_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    train_subset_size: Optional[int] = DEFAULT_TRAIN_SUBSET,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )
    if train_subset_size is not None and train_subset_size < len(trainset):
        indices = random.Random(seed).sample(range(len(trainset)), train_subset_size)
        trainset = Subset(trainset, indices)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = correct = 0.0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if training:
            loss.backward()
            optimizer.step()
        batch = labels.size(0)
        total_loss += loss.item() * batch
        correct += (outputs.argmax(1) == labels).sum().item()
        total += batch
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
        train_loss, train_acc = _run_epoch(
            model, trainloader, criterion, device, optimizer
        )
        test_loss, test_acc = _run_epoch(model, testloader, criterion, device)
        history.append(EpochMetrics(epoch, train_loss, train_acc, test_loss, test_acc))
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
    correct = torch.zeros(10, dtype=torch.long)
    total = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            for label, pred in zip(labels, preds):
                total[label] += 1
                correct[label] += int(label == pred)
    return {
        CIFAR10_CLASSES[i]: 100.0 * correct[i].item() / total[i].item()
        for i in range(10)
    }


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_images: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    images, labels, preds = [], [], []
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            images.append(batch_images.cpu())
            labels.append(batch_labels)
            preds.append(model(batch_images).argmax(1).cpu())
            if sum(x.size(0) for x in images) >= max_images:
                break
    images_cat = torch.cat(images)[:max_images]
    labels_cat = torch.cat(labels)[:max_images]
    preds_cat = torch.cat(preds)[:max_images]
    return images_cat, labels_cat, preds_cat
