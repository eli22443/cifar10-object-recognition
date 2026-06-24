"""Deconvolutional CIFAR-10 model for ADL HW4 Task 2–3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

from cifar_cnn import DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_MOMENTUM, EpochMetrics, Net


DEFAULT_LAMBDA = 0.1


@dataclass
class DeconvEpochMetrics:
    epoch: int
    train_loss: float
    train_ce: float
    train_rec: float
    train_acc: float
    test_loss: float
    test_ce: float
    test_rec: float
    test_acc: float


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-channel MSE, averaged over RGB channels."""
    channel_losses = [F.mse_loss(recon[:, c], target[:, c]) for c in range(3)]
    return sum(channel_losses) / 3.0


class DeconvNet(nn.Module):
    """Encoder–classifier–decoder network for classification and reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv1 = nn.ConvTranspose2d(6, 3, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        z1, idx1 = self.pool(x)
        x = F.relu(self.conv2(z1))
        z2, idx2 = self.pool(x)
        return z1, z2, idx1, idx2

    def classify(self, z2: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(z2, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def decode_from_z1(self, z1: torch.Tensor, idx1: torch.Tensor) -> torch.Tensor:
        x = self.unpool(z1, idx1)
        return F.relu(self.deconv1(x))

    def decode_from_z2(
        self,
        z2: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
    ) -> torch.Tensor:
        x = self.unpool(z2, idx2)
        x = F.relu(self.deconv2(x))
        return self.decode_from_z1(x, idx1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1, z2, idx1, idx2 = self.encode(x)
        logits = self.classify(z2)
        recon = self.decode_from_z2(z2, idx1, idx2)
        return logits, recon, z1, z2, idx1, idx2


def load_encoder_from_net(deconv_model: DeconvNet, classifier: Net) -> None:
    """Initialize encoder and classifier from a trained Task 1 model."""
    deconv_model.conv1.load_state_dict(classifier.conv1.state_dict())
    deconv_model.conv2.load_state_dict(classifier.conv2.state_dict())
    deconv_model.fc1.load_state_dict(classifier.fc1.state_dict())
    deconv_model.fc2.load_state_dict(classifier.fc2.state_dict())
    deconv_model.fc3.load_state_dict(classifier.fc3.state_dict())


def run_deconv_epoch(
    model: DeconvNet,
    loader: DataLoader,
    device: torch.device,
    lam: float,
    optimizer: Optional[Optimizer] = None,
) -> Tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_ce = 0.0
    total_rec = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits, recon, _, _, _, _ = model(images)
        loss_ce = ce_loss_fn(logits, labels)
        loss_rec = reconstruction_loss(recon, images)
        loss = loss_ce + lam * loss_rec

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_ce += loss_ce.item() * batch_size
        total_rec += loss_rec.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, total_ce / total, total_rec / total, correct / total


def train_deconv_model(
    model: DeconvNet,
    trainloader: DataLoader,
    testloader: DataLoader,
    device: torch.device,
    lam: float = DEFAULT_LAMBDA,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    momentum: float = DEFAULT_MOMENTUM,
) -> List[DeconvEpochMetrics]:
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    history: List[DeconvEpochMetrics] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_ce, train_rec, train_acc = run_deconv_epoch(
            model, trainloader, device, lam, optimizer=optimizer
        )
        test_loss, test_ce, test_rec, test_acc = run_deconv_epoch(
            model, testloader, device, lam
        )
        metrics = DeconvEpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_ce=train_ce,
            train_rec=train_rec,
            train_acc=train_acc,
            test_loss=test_loss,
            test_ce=test_ce,
            test_rec=test_rec,
            test_acc=test_acc,
        )
        history.append(metrics)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train acc {100 * train_acc:.1f}% | test acc {100 * test_acc:.1f}% | "
            f"train rec {train_rec:.4f} | test rec {test_rec:.4f}"
        )

    return history


def collect_reconstructions(
    model: DeconvNet,
    loader: DataLoader,
    device: torch.device,
    max_images: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    originals: List[torch.Tensor] = []
    recons: List[torch.Tensor] = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _, recon, _, _, _, _ = model(images)
            originals.append(images.cpu())
            recons.append(recon.cpu())
            if sum(batch.size(0) for batch in originals) >= max_images:
                break

    return torch.cat(originals)[:max_images], torch.cat(recons)[:max_images]
