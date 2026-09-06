# Object Recognition on CIFAR-10

Applied Deep Learning coursework — CNN classifier with deconvolutional reconstruction.

This project trains a convolutional neural network for object recognition on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck), then extends the model with a deconvolutional decoder for image reconstruction and latent-feature analysis.

## What’s included

| Phase | Description |
|-------|-------------|
| **1** | Train a CNN classifier on a 6,000-image training subset |
| **2** | Add a deconvolutional decoder; jointly optimize classification and reconstruction |
| **3** | Analyze reconstructions and latent features from the trained encoder–decoder model |

**Source code**

- [`report.ipynb`](report.ipynb) — full experiment notebook
- [`cifar_cnn.py`](cifar_cnn.py) — CIFAR-10 data loading, CNN (`Net`), training utilities
- [`deconv_net.py`](deconv_net.py) — `DeconvNet` encoder–classifier–decoder and training helpers

## Requirements

- **Python 3.12+**
- See [`requirements.txt`](requirements.txt) for packages (`torch`, `torchvision`, `numpy`, `matplotlib`, Jupyter)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

1. Open [`report.ipynb`](report.ipynb) in Jupyter or VS Code / Cursor.
2. Run the cells in order.

On first run, CIFAR-10 downloads automatically into `./data` (ignored by git). Training uses CPU or CUDA if available.

```bash
jupyter notebook report.ipynb
```

## License / academic use

Coursework for Applied Deep Learning. Reuse for learning is fine; do not submit it as your own work.
