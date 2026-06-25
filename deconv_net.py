"""Deconvolutional CIFAR-10 model for ADL HW4 Task 2–3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

from cifar_cnn import DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_MOMENTUM, Net, denormalize

DEFAULT_LAMBDA = 0.1
ENCODER_KEYS = ("conv1", "conv2", "fc1", "fc2", "fc3")


@dataclass
class DeconvEpochMetrics:
    epoch: int
    train_acc: float
    test_acc: float
    train_rec: float
    test_rec: float


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.stack([F.mse_loss(recon[:, c], target[:, c]) for c in range(3)]).mean()


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
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1, idx1 = self.pool(F.relu(self.conv1(x)))
        z2, idx2 = self.pool(F.relu(self.conv2(z1)))
        return z1, z2, idx1, idx2

    def classify(self, z2: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(torch.flatten(z2, 1)))
        return self.fc3(F.relu(self.fc2(x)))

    def decode_from_z1(self, z1: torch.Tensor, idx1: torch.Tensor) -> torch.Tensor:
        return F.relu(self.deconv1(self.unpool(z1, idx1)))

    def decode_from_z2(
        self, z2: torch.Tensor, idx1: torch.Tensor, idx2: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode_from_z1(F.relu(self.deconv2(self.unpool(z2, idx2))), idx1)

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1, z2, idx1, idx2 = self.encode(x)
        return self.classify(z2), self.decode_from_z2(z2, idx1, idx2), z1, z2, idx1, idx2


def load_encoder_from_net(deconv_model: DeconvNet, classifier: Net) -> None:
    for name in ENCODER_KEYS:
        getattr(deconv_model, name).load_state_dict(getattr(classifier, name).state_dict())


def _run_deconv_epoch(
    model: DeconvNet,
    loader: DataLoader,
    device: torch.device,
    lam: float,
    optimizer=None,
) -> Tuple[float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    ce_fn = nn.CrossEntropyLoss()
    total_loss = total_ce = total_rec = correct = 0.0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        logits, recon, *_ = model(images)
        loss_ce = ce_fn(logits, labels)
        loss_rec = reconstruction_loss(recon, images)
        loss = loss_ce + lam * loss_rec
        if training:
            loss.backward()
            optimizer.step()
        batch = labels.size(0)
        total += batch
        total_loss += loss.item() * batch
        total_ce += loss_ce.item() * batch
        total_rec += loss_rec.item() * batch
        correct += (logits.argmax(1) == labels).sum().item()
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
        _, _, train_rec, train_acc = _run_deconv_epoch(
            model, trainloader, device, lam, optimizer,
        )
        _, _, test_rec, test_acc = _run_deconv_epoch(model, testloader, device, lam)
        history.append(DeconvEpochMetrics(epoch, train_acc, test_acc, train_rec, test_rec))
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train acc {100 * train_acc:.1f}% | test acc {100 * test_acc:.1f}% | "
                f"train rec {train_rec:.4f} | test rec {test_rec:.4f}"
            )
    return history


def _collect_batches(model: DeconvNet, loader: DataLoader, device: torch.device, max_images: int):
    model.eval()
    originals, recons = [], []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _, recon, *_ = model(images)
            originals.append(images.cpu())
            recons.append(recon.cpu())
            if sum(x.size(0) for x in originals) >= max_images:
                break
    return torch.cat(originals)[:max_images], torch.cat(recons)[:max_images]


def collect_reconstructions(
    model: DeconvNet, loader: DataLoader, device: torch.device, max_images: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _collect_batches(model, loader, device, max_images)


def get_sample_image(loader: DataLoader, index: int = 0) -> Tuple[torch.Tensor, int]:
    return loader.dataset[index]


def _mask_channel(features: torch.Tensor, channel: int) -> torch.Tensor:
    masked = torch.zeros_like(features)
    masked[:, channel] = features[:, channel]
    return masked


@torch.no_grad()
def encode_image(
    model: DeconvNet, image: torch.Tensor, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    _, _, z1, z2, idx1, idx2 = model(image)
    return image.cpu(), z1, z2, idx1, idx2


@torch.no_grad()
def reconstruct_z1_channels(
    model: DeconvNet, z1: torch.Tensor, idx1: torch.Tensor,
) -> List[torch.Tensor]:
    model.eval()
    return [
        model.decode_from_z1(_mask_channel(z1, c), idx1).cpu()
        for c in range(z1.shape[1])
    ]


@torch.no_grad()
def reconstruct_z2_channels(
    model: DeconvNet,
    z2: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    channels: Sequence[int],
) -> List[torch.Tensor]:
    model.eval()
    return [
        model.decode_from_z2(_mask_channel(z2, c), idx1, idx2).cpu()
        for c in channels
    ]


def plot_channel_ablation(
    original: torch.Tensor,
    reconstructions: Sequence[torch.Tensor],
    channel_labels: Sequence[str],
    title: str,
) -> None:
    fig, axes_raw = plt.subplots(1, len(reconstructions) + 1, figsize=(2 * (len(reconstructions) + 1), 2.2))
    figure = cast(Figure, fig)
    axes = cast(List[Axes], list(np.atleast_1d(axes_raw).ravel()))
    panels = [(original, "Original")] + list(zip(reconstructions, channel_labels))
    for ax, (tensor, label) in zip(axes, panels):
        img = tensor[0] if tensor.dim() == 4 else tensor
        ax.imshow(np.clip(denormalize(img).numpy().transpose(1, 2, 0), 0, 1))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    plt.show()
