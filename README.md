# Hopf Autoresearch

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MSAI Research](https://img.shields.io/badge/MSAI-Northeastern-c41e3a.svg)](https://www.khoury.northeastern.edu/program/msai/)

**Hopf Latent Spaces** — nested complex Hopf fibrations as a geometric inductive bias for neural network representations.

Fork of [Karpathy autoresearch](https://github.com/karpathy/autoresearch) / [autoresearch-macos](https://github.com/miolini/autoresearch-macos), extended with differentiable Hopf projectors, structured-task benchmarks, and a full MSAI research paper.

> **Research question:** Can latent representations improve when projected through Hopf fibration geometry ($S^1 \hookrightarrow S^{3} \twoheadrightarrow S^{2}$) instead of unconstrained Euclidean space?

**Paper (PDF):** [`paper/hopf_latent_spaces.pdf`](paper/hopf_latent_spaces.pdf)  
**Author:** Joseph Defendre · MSAI, Khoury College · [defendre.j@northeastern.edu](mailto:defendre.j@northeastern.edu)

---

## Key findings (so far)

| Domain | Result |
|--------|--------|
| **Periodic regression** (3 seeds) | Hopf **−6.2%** RMSE vs Euclidean |
| **LM pretraining** (2 seeds) | Per-block Hopf α=0.03 **−0.7%** val BPB |
| **MNIST AE** | Lower fiber-phase variance (more structured latents) |
| **Input Hopf / strong α / hypersphere LM** | Consistently **worse** — reported as negative results |

Hopf works as a **light per-block nudge**, not a hard input constraint. See the paper for hypotheses, methodology, and skeptic analysis.

---

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), Apple Silicon (MPS) or NVIDIA GPU.

```bash
git clone https://github.com/jchacker5/hopf-autoresearch.git
cd hopf-autoresearch

uv sync
uv run prepare.py --num-shards 4    # ~2 min; downloads data + tokenizer
uv run train.py                     # ~5 min training run
```

**Winning LM config** (in `train.py`):

```python
USE_HOPF_LATENT = True
HOPF_NUM_LEVELS = 2
HOPF_MIX_ALPHA = 0.03      # multi-seed stable winner
HOPF_PLACEMENT = "per_block"
```

---

## Project structure

```
train.py              # GPT + HopfLatentProjector (main experiment file)
hopf_layers.py        # Reusable Hopf / hypersphere modules
prepare.py              # Data, tokenizer, eval harness (fixed)
program.md              # Autonomous agent research loop instructions

paper/                  # LaTeX thesis paper + PDF + figures
eval/                   # Structured synthetic tasks (periodic, hierarchy, modular)
experiments/            # MNIST AE, multi-seed LM runners, result JSON

theory/                 # Early theory notes
results.tsv             # Local experiment log (gitignored)
```

---

The 0.7% LM result is exploratory: it is from two five-minute seeds after configuration search, on validation text. It does not establish a causal contribution from the Hopf map, a large-scale language-model advantage, or a physical mechanism. See the [skeptic review](paper/SKEPTIC_REVIEW.md) for the matched-compute controls and independent data needed for confirmation.

## Run all benchmarks

```bash
# Structured tasks (periodic / hierarchical / modular) — 3 seeds
uv run python eval/structured_tasks.py

# MNIST autoencoder comparison
uv run python experiments/ae_mnist.py

# Multi-seed LM sweep (long — ~2h for full matrix)
uv run python experiments/publication_lm.py

# Regenerate paper figures
uv run python paper/figures/generate_figures.py
cd paper && make pdf
```

---

## Autonomous overnight research

Point your AI agent at [`program.md`](program.md):

```
Read program.md and kick off a new experiment. Setup first.
```

The agent edits `train.py`, trains 5 minutes, logs to `results.tsv`, keeps or discards commits. Expect ~12 experiments/hour.

```bash
git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')
```

---

## Contribute — we want your help

This is active MSAI research. **Contributions are welcome and encouraged**, especially:

| Priority | Contribution idea |
|----------|-------------------|
| High | **Third seed** + statistical tests on LM results |
| High | **Harder structured tasks** (non-ceiling modular arithmetic, deep trees) |
| High | **CUDA / Linux** fork with FlashAttention (upstream autoresearch parity) |
| Medium | Auxiliary geometric loss $\mathcal{L}_{\mathrm{fib}}$ |
| Medium | Persistent homology / topology metrics on learned latents |
| Medium | Arithmetic & algorithmic exact-match eval harness |
| Low | Docs, typos, figure polish, Colab notebook |

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup, branch naming, and PR guidelines.

**Open issues we need help with:**
- Replicate α=0.03 result on NVIDIA GPU (H100/A100)
- Does Hopf help on TinyStories or math-heavy corpora?
- Per-block vs per-attention-head placement ablations
- Quaternion normalization ablation (is Hopf map necessary vs normalize-only?)

Open a [GitHub Issue](https://github.com/jchacker5/hopf-autoresearch/issues) before large changes, or jump in with a small PR.

---

## Citation

```bibtex
@misc{defendre2026hopf,
  author  = {Joseph Defendre},
  title   = {Hopf Latent Spaces: Nested Complex Hopf Fibrations as Geometric Inductive Bias for Structured Latent Representations in Neural Networks},
  year    = {2026},
  school  = {Northeastern University, Khoury College of Computer Sciences},
  note    = {MSAI Research Paper},
  url     = {https://github.com/jchacker5/hopf-autoresearch}
}
```

---

## Acknowledgments

Built on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos). Inspired by classical Hopf fibration geometry and geometric deep learning.

## License

MIT — see [LICENSE](LICENSE).
