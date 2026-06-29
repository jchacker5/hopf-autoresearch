"""MNIST conv autoencoder: baseline vs Hopf vs hypersphere latent geometry."""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hopf_layers import (
    HopfLatentProjector,
    HypersphereProjector,
    hopf_fiber_phase_variance,
    latent_fiber_phase_variance,
)

LATENT_DIM = 64
HOPF_LEVELS = 2
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
SEED = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ConvAE(nn.Module):
    def __init__(self, mode="baseline", hopf_alpha=0.05, hopf_levels=HOPF_LEVELS):
        super().__init__()
        self.mode = mode
        self.hopf_alpha = hopf_alpha
        self.hopf_levels = hopf_levels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        self.hopf = (
            HopfLatentProjector(LATENT_DIM, hopf_levels)
            if mode == "hopf"
            else None
        )
        self.hypersphere = (
            HypersphereProjector(LATENT_DIM)
            if mode == "hypersphere"
            else None
        )

    def apply_latent_geometry(self, z):
        if self.hopf is not None:
            return self.hopf_alpha * self.hopf(z) + (1.0 - self.hopf_alpha) * z
        if self.hypersphere is not None:
            return self.hypersphere(z)
        return z

    def forward(self, x):
        z_raw = self.encoder(x)
        z = self.apply_latent_geometry(z_raw)
        recon = self.decoder(z)
        return recon, z_raw, z


def init_hopf_projector(model: ConvAE):
    if model.hopf is None:
        return
    s = 0.02
    nn.init.uniform_(model.hopf.in_proj.weight, -s, s)
    nn.init.zeros_(model.hopf.out_proj.weight)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_phase_var = 0.0
    n_batches = 0

    for images, _ in loader:
        images = images.to(device)
        recon, _, z = model(images)
        total_mse += F.mse_loss(recon, images, reduction="mean").item()
        if model.hopf is not None:
            total_phase_var += hopf_fiber_phase_variance(z, model.hopf)
        else:
            total_phase_var += latent_fiber_phase_variance(z, model.hopf_levels)
        n_batches += 1

    return {
        "reconstruction_mse": total_mse / max(n_batches, 1),
        "fiber_phase_variance": total_phase_var / max(n_batches, 1),
    }


def train_variant(name, mode, device, train_loader, test_loader, hopf_alpha=0.05):
    torch.manual_seed(SEED)
    if device.type == "mps":
        torch.mps.manual_seed(SEED)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model = ConvAE(mode=mode, hopf_alpha=hopf_alpha).to(device)
    init_hopf_projector(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epoch_losses = []
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, _ in train_loader:
            images = images.to(device)
            recon, _, _ = model(images)
            loss = F.mse_loss(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        epoch_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(epoch_loss)
        print(f"  [{name}] epoch {epoch + 1}/{EPOCHS} loss={epoch_loss:.6f}")

    metrics = evaluate(model, test_loader, device)
    return {
        "name": name,
        "mode": mode,
        "hopf_alpha": hopf_alpha if mode == "hopf" else None,
        "hopf_levels": HOPF_LEVELS if mode == "hopf" else None,
        "training_loss_per_epoch": epoch_losses,
        "final_training_loss": epoch_losses[-1],
        **metrics,
    }


def main():
    device = get_device()
    print(f"Device: {device}")

    transform = transforms.ToTensor()
    data_dir = ROOT / "data" / "mnist"
    train_set = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory
    )

    variants = [
        ("baseline", "baseline", None),
        ("hopf_per_block", "hopf", 0.05),
        ("hopf_alpha_0.15", "hopf", 0.15),
        ("hypersphere", "hypersphere", None),
    ]

    results = {
        "config": {
            "latent_dim": LATENT_DIM,
            "hopf_levels": HOPF_LEVELS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
            "device": str(device),
        },
        "variants": [],
    }

    for name, mode, alpha in variants:
        print(f"\nTraining variant: {name}")
        hopf_alpha = alpha if alpha is not None else 0.05
        result = train_variant(name, mode, device, train_loader, test_loader, hopf_alpha=hopf_alpha)
        results["variants"].append(result)

    out_path = ROOT / "experiments" / "results_ae_mnist.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")

    print("\n=== Summary ===")
    for v in results["variants"]:
        print(
            f"{v['name']:20s}  "
            f"train_loss={v['final_training_loss']:.6f}  "
            f"recon_mse={v['reconstruction_mse']:.6f}  "
            f"phase_var={v['fiber_phase_variance']:.6f}"
        )


if __name__ == "__main__":
    main()