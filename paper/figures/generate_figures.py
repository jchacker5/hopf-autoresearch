#!/usr/bin/env python3
"""Generate publication figures for Hopf Latent Spaces paper."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
PAPER = ROOT.parent
REPO = PAPER.parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig_hopf_schematic():
    """Hopf fibration: S^1 fiber over S^2 base (schematic)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.2, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Base S^2 (circle projection)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5, label=r"Base $S^2$")
    ax.fill(np.cos(theta) * 0.35, np.sin(theta) * 0.35, color="#4C72B0", alpha=0.15)
    ax.text(0, -0.55, r"Base $b \in S^2$", ha="center", fontsize=8)

    # Linked fibers (circles in 2D schematic)
    for i, ang in enumerate([0.4, 2.0, 3.8, 5.2]):
        cx, cy = 0.85 * np.cos(ang), 0.85 * np.sin(ang)
        r = 0.22
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="#C44E52", lw=1.2, ls="--"))
        if i == 0:
            ax.text(cx, cy + r + 0.08, r"$S^1$ fiber", ha="center", fontsize=7, color="#C44E52")

    ax.annotate("", xy=(0.5, 0.5), xytext=(0.85, 0.0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.text(0.55, 0.35, r"$\pi: S^3 \to S^2$", fontsize=8, color="gray")
    ax.set_title("Hopf Fibration (schematic)", fontweight="bold", pad=6)
    fig.savefig(ROOT / "fig1_hopf_schematic.pdf")
    fig.savefig(ROOT / "fig1_hopf_schematic.png")
    plt.close(fig)


def fig_architecture():
    """Transformer block with per-block Hopf projector."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (1, 4.5, 8, 0.9, "Token Embedding", "#DDDDDD"),
        (1, 3.2, 8, 1.0, "Transformer Block $i$", "#E8F4FC"),
        (1.3, 3.35, 3.2, 0.7, "Hopf Projector\n($\\alpha$-blend)", "#FCE8E6"),
        (5.0, 3.35, 3.5, 0.7, "Attention + MLP", "#E8F4FC"),
        (1, 1.8, 8, 0.9, "LM Head", "#DDDDDD"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", lw=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7)

    for y1, y2 in [(4.5, 4.2), (3.2, 2.7), (1.8, 1.5)]:
        ax.annotate("", xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(5, 0.6, r"$x' = \alpha\,\mathrm{Hopf}(x) + (1-\alpha)\,x$", ha="center", fontsize=8)
    ax.set_title("Per-Block Hopf Placement", fontweight="bold")
    fig.savefig(ROOT / "fig2_architecture.pdf")
    fig.savefig(ROOT / "fig2_architecture.png")
    plt.close(fig)


def fig_lm_multiseed():
    data = json.loads((REPO / "experiments/results_lm_multiseed.json").read_text())["summary"]
    names = ["baseline", "hopf_alpha_0.03", "hopf_winner", "hopf_input", "hopf_alpha_0.08", "hypersphere"]
    labels = ["Euclidean", r"Hopf $\alpha{=}0.03$", r"Hopf $\alpha{=}0.05$",
              "Hopf input", r"Hopf $\alpha{=}0.08$", "Hypersphere"]
    means = [data[n]["mean"] for n in names]
    stds = [data[n]["std"] for n in names]
    colors = ["#4C72B0", "#55A868", "#55A868", "#CCB974", "#C44E52", "#C44E52"]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", lw=0.5, alpha=0.85)
    ax.axhline(means[0], color="#4C72B0", ls=":", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Validation BPB $\\downarrow$")
    ax.set_title("LM Pretraining (2 seeds)", fontweight="bold")
    ax.set_ylim(1.72, 1.82)
    fig.savefig(ROOT / "fig3_lm_multiseed.pdf")
    fig.savefig(ROOT / "fig3_lm_multiseed.png")
    plt.close(fig)


def fig_structured_tasks():
    data = json.loads((REPO / "eval/results_structured.json").read_text())["summary"]
    tasks = ["periodic", "hierarchical"]
    methods = ["baseline", "hopf", "hypersphere"]
    mlabels = ["Euclidean", "Hopf", "Hypersphere"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, task in zip(axes, tasks):
        s = data[task]
        if task == "periodic":
            vals = [s[m]["mean"] for m in methods]
            errs = [s[m]["std"] for m in methods]
            ax.bar(mlabels, vals, yerr=errs, capsize=3, color=["#4C72B0", "#55A868", "#CCB974"],
                   edgecolor="k", lw=0.5)
            ax.set_ylabel("RMSE $\\downarrow$")
            ax.set_title("Periodic Regression", fontweight="bold")
        else:
            vals = [s[m]["mean"] * 100 for m in methods]
            errs = [s[m]["std"] * 100 for m in methods]
            ax.bar(mlabels, vals, yerr=errs, capsize=3, color=["#4C72B0", "#55A868", "#CCB974"],
                   edgecolor="k", lw=0.5)
            ax.set_ylabel("Accuracy (%) $\\uparrow$")
            ax.set_ylim(98, 100)
            ax.set_title("Hierarchical Classification", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "fig4_structured_tasks.pdf")
    fig.savefig(ROOT / "fig4_structured_tasks.png")
    plt.close(fig)


def fig_mnist_ae():
    data = json.loads((REPO / "experiments/results_ae_mnist.json").read_text())["variants"]
    names = [v["name"] for v in data]
    mse = [v["reconstruction_mse"] for v in data]
    phase = [v["fiber_phase_variance"] for v in data]

    fig, ax1 = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(names))
    ax1.bar(x - 0.2, mse, 0.4, label="Recon MSE", color="#4C72B0", edgecolor="k", lw=0.5)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, phase, 0.4, label="Phase variance", color="#C44E52", edgecolor="k", lw=0.5, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Baseline", "Hopf 0.05", "Hopf 0.15", "Hypersphere"], rotation=25, ha="right", fontsize=7)
    ax1.set_ylabel("MSE $\\downarrow$")
    ax2.set_ylabel("Fiber phase var.")
    ax1.set_title("MNIST Autoencoder", fontweight="bold")
    fig.savefig(ROOT / "fig5_mnist_ae.pdf")
    fig.savefig(ROOT / "fig5_mnist_ae.png")
    plt.close(fig)


def fig_alpha_sweep():
    """Autoresearch + multiseed alpha sensitivity."""
    alphas = [0.03, 0.05, 0.08, 0.15, 0.25, 0.5]
    # multiseed for 0.03, 0.05, 0.08; single-seed autoresearch for others from results.tsv
    bpbs = [1.7384, 1.7440, 1.7590, 1.8608, 1.8476, 1.8142]
    baseline = 1.7502

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(alphas, bpbs, "o-", color="#55A868", lw=1.5, markersize=5, label="Hopf per-block")
    ax.axhline(baseline, color="#4C72B0", ls="--", lw=1.2, label="Euclidean baseline")
    ax.set_xlabel("Mix weight $\\alpha$")
    ax.set_ylabel("val BPB $\\downarrow$")
    ax.set_xscale("log")
    ax.set_title("$\\alpha$ Sensitivity (per-block)", fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9)
    fig.savefig(ROOT / "fig6_alpha_sweep.pdf")
    fig.savefig(ROOT / "fig6_alpha_sweep.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_hopf_schematic()
    fig_architecture()
    fig_lm_multiseed()
    fig_structured_tasks()
    fig_mnist_ae()
    fig_alpha_sweep()
    print("Figures written to", ROOT)