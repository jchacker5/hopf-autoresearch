"""
Structured synthetic tasks for Hopf Latent Spaces evaluation.

Tasks:
  1. Periodic regression: y = sin(phase * freq) from noisy phase input
  2. Hierarchical labels: tree-structured multi-class classification
  3. Modular arithmetic: predict (a + b) mod p from encoded pair

Compares baseline vs hopf (per_block-style hidden projection) vs hypersphere
over 3 seeds. Results saved to eval/results_structured.json.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopf_layers import HopfLatentProjector, HypersphereProjector

Method = Literal["baseline", "hopf", "hypersphere"]
TaskName = Literal["periodic", "hierarchical", "modular"]

HIDDEN_DIM = 96
HOPF_ALPHA = 0.5
HOPF_LEVELS = 2
TRAIN_SIZE = 10_000
TEST_SIZE = 2_000
BATCH_SIZE = 256
EPOCHS = 300
LR = 1e-3
SEEDS = [0, 1, 2]
METHODS: list[Method] = ["baseline", "hopf", "hypersphere"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def make_periodic_dataset(
    n: int,
    freqs: list[float],
    noise_std: float = 0.15,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Input: [noisy cos, noisy sin, freq_norm]; target: sin(phase * freq)."""
    g = torch.Generator().manual_seed(seed)
    freq_idx = torch.randint(0, len(freqs), (n,), generator=g)
    freq = torch.tensor(freqs, dtype=torch.float32)[freq_idx]
    phase = torch.rand(n, generator=g) * 2 * math.pi
    target = torch.sin(phase * freq)
    cos_p = torch.cos(phase) + noise_std * torch.randn(n, generator=g)
    sin_p = torch.sin(phase) + noise_std * torch.randn(n, generator=g)
    freq_norm = (freq - min(freqs)) / (max(freqs) - min(freqs) + 1e-8)
    x = torch.stack([cos_p, sin_p, freq_norm], dim=1)
    return x, target.unsqueeze(1)


@dataclass
class HierarchySpec:
    num_leaves: int
    depth: int
    parent_of: list[int]
    leaf_to_class: list[int]


def build_hierarchy_tree(branching: tuple[int, ...] = (3, 3)) -> HierarchySpec:
    """Binary/ternary tree: root id 0, internal nodes, leaves are classes."""
    parent_of = [-1]
    next_id = 1
    frontier = [0]
    depth = 0
    while depth < len(branching):
        b = branching[depth]
        new_frontier = []
        for node in frontier:
            for _ in range(b):
                parent_of.append(node)
                new_frontier.append(next_id)
                next_id += 1
        frontier = new_frontier
        depth += 1
    leaf_ids = frontier
    leaf_to_class = list(range(len(leaf_ids)))
    return HierarchySpec(
        num_leaves=len(leaf_ids),
        depth=len(branching),
        parent_of=parent_of,
        leaf_to_class=leaf_to_class,
    )


def hierarchy_path_features(leaf_id: int, spec: HierarchySpec, noise_std: float, g: torch.Generator) -> torch.Tensor:
    """Feature vector: one-hot path from root to leaf + noise."""
    path = []
    node = leaf_id
    while node >= 0:
        path.append(node)
        node = spec.parent_of[node]
    path = list(reversed(path))
    feat_dim = len(spec.parent_of)
    x = torch.zeros(feat_dim)
    for node in path:
        x[node] = 1.0
    x += noise_std * torch.randn(feat_dim, generator=g)
    return x


def make_hierarchical_dataset(
    n: int,
    spec: HierarchySpec,
    noise_std: float = 0.25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    num_leaves = spec.num_leaves
    # Leaf ids are the last num_leaves node indices
    first_leaf = len(spec.parent_of) - num_leaves
    leaf_ids = list(range(first_leaf, len(spec.parent_of)))
    labels = torch.randint(0, num_leaves, (n,), generator=g)
    xs = []
    for i in range(n):
        leaf_id = leaf_ids[labels[i].item()]
        xs.append(hierarchy_path_features(leaf_id, spec, noise_std, g))
    x = torch.stack(xs)
    return x, labels


def make_modular_dataset(
    n: int,
    p: int = 11,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, p, (n,), generator=g)
    b = torch.randint(0, p, (n,), generator=g)
    target = (a + b) % p
    x = F.one_hot(a, p).float()
    x = torch.cat([x, F.one_hot(b, p).float()], dim=1)
    return x, target


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StructuredMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        method: Method,
        task: TaskName,
        hidden_dim: int = HIDDEN_DIM,
        hopf_alpha: float = HOPF_ALPHA,
        hopf_levels: int = HOPF_LEVELS,
    ):
        super().__init__()
        self.method = method
        self.task = task
        self.hopf_alpha = hopf_alpha
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        if method == "hopf":
            self.projector: nn.Module | None = HopfLatentProjector(hidden_dim, num_levels=hopf_levels)
        elif method == "hypersphere":
            self.projector = HypersphereProjector(hidden_dim)
        else:
            self.projector = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        if self.projector is not None:
            if self.method == "hopf":
                h = self.hopf_alpha * self.projector(h) + (1.0 - self.hopf_alpha) * h
            else:
                h = self.projector(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        if yb.dtype in (torch.int32, torch.int64):
            loss = criterion(out, yb)
        else:
            loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task: TaskName,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    if task == "periodic":
        mse_sum = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mse_sum += F.mse_loss(pred, yb, reduction="sum").item()
            n += yb.size(0)
        rmse = math.sqrt(mse_sum / max(n, 1))
        return {"rmse": rmse, "metric": rmse, "metric_name": "rmse", "lower_is_better": True}

    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    acc = correct / max(total, 1)
    return {"accuracy": acc, "metric": acc, "metric_name": "accuracy", "lower_is_better": False}


def run_single(
    task: TaskName,
    method: Method,
    seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)

    freqs = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0]
    hierarchy = build_hierarchy_tree((3, 3))
    modulus = 11

    if task == "periodic":
        x_train, y_train = make_periodic_dataset(TRAIN_SIZE, freqs, seed=seed)
        x_test, y_test = make_periodic_dataset(TEST_SIZE, freqs, seed=seed + 10_000)
        input_dim, output_dim = 3, 1
    elif task == "hierarchical":
        x_train, y_train = make_hierarchical_dataset(TRAIN_SIZE, hierarchy, seed=seed)
        x_test, y_test = make_hierarchical_dataset(TEST_SIZE, hierarchy, seed=seed + 10_000)
        input_dim = x_train.size(1)
        output_dim = hierarchy.num_leaves
    else:
        x_train, y_train = make_modular_dataset(TRAIN_SIZE, p=modulus, seed=seed)
        x_test, y_test = make_modular_dataset(TEST_SIZE, p=modulus, seed=seed + 10_000)
        input_dim, output_dim = 2 * modulus, modulus

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

    model = StructuredMLP(input_dim, output_dim, method=method, task=task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if task == "periodic":
        criterion: Callable = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, device)

    metrics = evaluate(model, test_loader, task, device)
    return {
        "task": task,
        "method": method,
        "seed": seed,
        **metrics,
    }


def summarize_runs(runs: list[dict]) -> dict:
    summary: dict[str, dict[str, dict]] = {}
    for task in ("periodic", "hierarchical", "modular"):
        summary[task] = {}
        for method in METHODS:
            task_runs = [r for r in runs if r["task"] == task and r["method"] == method]
            values = [r["metric"] for r in task_runs]
            mean = sum(values) / len(values)
            if len(values) > 1:
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            metric_name = task_runs[0]["metric_name"]
            lower_is_better = task_runs[0]["lower_is_better"]
            summary[task][method] = {
                "mean": mean,
                "std": std,
                "metric_name": metric_name,
                "lower_is_better": lower_is_better,
                "n_seeds": len(values),
            }
    return summary


def print_summary_table(summary: dict) -> None:
    print("\n" + "=" * 72)
    print("STRUCTURED TASKS SUMMARY (mean ± std over 3 seeds)")
    print("=" * 72)
    header = f"{'Task':<14} {'Metric':<10} " + " ".join(f"{m:<22}" for m in METHODS)
    print(header)
    print("-" * 72)
    for task in ("periodic", "hierarchical", "modular"):
        metric_name = summary[task]["baseline"]["metric_name"]
        cells = []
        for method in METHODS:
            s = summary[task][method]
            cells.append(f"{s['mean']:.4f}±{s['std']:.4f}")
        print(f"{task:<14} {metric_name:<10} " + " ".join(f"{c:<22}" for c in cells))
    print("=" * 72)


def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"Hidden dim: {HIDDEN_DIM}, epochs: {EPOCHS}, seeds: {SEEDS}")

    runs: list[dict] = []
    for task in ("periodic", "hierarchical", "modular"):
        for method in METHODS:
            for seed in SEEDS:
                print(f"Running task={task} method={method} seed={seed} ...", flush=True)
                result = run_single(task, method, seed, device)
                runs.append(result)
                print(
                    f"  -> {result['metric_name']}={result['metric']:.4f}",
                    flush=True,
                )

    summary = summarize_runs(runs)
    print_summary_table(summary)

    out_path = Path(__file__).resolve().parent / "results_structured.json"
    payload = {
        "config": {
            "hidden_dim": HIDDEN_DIM,
            "hopf_alpha": HOPF_ALPHA,
            "hopf_levels": HOPF_LEVELS,
            "train_size": TRAIN_SIZE,
            "test_size": TEST_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "seeds": SEEDS,
            "methods": METHODS,
        },
        "runs": runs,
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()