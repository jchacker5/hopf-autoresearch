# hopf-autoresearch

Autonomous research repo for testing [**Hopf Latent Spaces**](theory/HOPF_LATENT_SPACES.md) — nested complex Hopf fibrations as a topological inductive bias in language model pretraining.

Built on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) / [autoresearch-macos](https://github.com/miolini/autoresearch-macos) loop: an AI agent edits `train.py`, trains for 5 minutes, keeps improvements, repeats.

## Hypothesis

Projecting token embeddings through Hopf fibrations \( S^1 \hookrightarrow S^{2n+1} \twoheadrightarrow \mathbb{CP}^n \) gives latent spaces built-in **hierarchy**, **periodicity**, and **topological linking** — improving `val_bpb` vs. Euclidean and hyperspherical baselines.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), Apple Silicon (MPS) or NVIDIA GPU.

```bash
cd hopf-autoresearch

# Install dependencies
uv sync

# Download data + train tokenizer (~2 min, use --num-shards 4 for quick Mac test)
uv run prepare.py --num-shards 4

# Baseline run (~5 min)
uv run train.py

# Autonomous research — point your agent at program.md
# "Read program.md and kick off a new experiment. Setup first."
```

## Project structure

```
prepare.py                    — data, tokenizer, eval harness (fixed)
train.py                      — GPT + HopfLatentProjector (agent edits this)
program.md                    — autonomous research instructions
theory/HOPF_LATENT_SPACES.md  — paper draft + math
results.tsv                   — experiment log (gitignored, created by agent)
```

## Hopf knobs (in train.py)

| Flag | Default | Description |
|------|---------|-------------|
| `USE_HOPF_LATENT` | `False` | Enable Hopf fibration projector |
| `HOPF_NUM_LEVELS` | `2` | Nested fibration depth |
| `HOPF_MIX_ALPHA` | `0.5` | Blend: α·Hopf + (1-α)·residual |
| `HOPF_PLACEMENT` | `"input"` | `"input"` or `"per_block"` |
| `USE_HYPERSPHERE_LATENT` | `False` | Ablation: unit-sphere only |

## Running overnight research

1. Create branch: `git checkout -b autoresearch/jun27`
2. Start agent with `program.md` as the skill
3. Agent loops: edit → train → log → keep/discard
4. Review `results.tsv` and git log in the morning

## Author

Joseph Defendre — MSAI, Northeastern University  
j.defendre@northeastern.edu

## License

MIT (inherits from autoresearch)