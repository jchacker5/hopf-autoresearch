# hopf-autoresearch

Autonomous experiments to test **Hopf Latent Spaces** as a topological inductive bias in language model pretraining.

Read `theory/HOPF_LATENT_SPACES.md` for the full hypothesis, math, and experiment roadmap before changing architecture.

## Research hypothesis

Standard Euclidean (or simple spherical) latent spaces lack built-in hierarchy, periodicity, and topological linking. **Hopf Latent Spaces** project embeddings onto nested circle fibers over complex projective bases via the Hopf fibration \( S^1 \hookrightarrow S^{2n+1} \twoheadrightarrow \mathbb{CP}^n \).

**Primary prediction:** enabling `USE_HOPF_LATENT` in `train.py` lowers `val_bpb` vs. the Euclidean baseline, especially on structured/algorithmic patterns in the training distribution.

**Secondary predictions to probe:**
- Deeper nesting (`HOPF_NUM_LEVELS` 1→2→3) helps hierarchy-sensitive patterns
- Higher `HOPF_MIX_ALPHA` (more Hopf, less residual) has a sweet spot — too high may hurt
- Per-layer Hopf (inside blocks) vs. input-only Hopf — compare both
- Hyperspherical baseline (`USE_HYPERSPHERE_LATENT`) as ablation control

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun27`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files**:
   - `README.md` — repo overview
   - `theory/HOPF_LATENT_SPACES.md` — paper draft + implementation notes
   - `prepare.py` — fixed constants, data prep, evaluation. **Do not modify.**
   - `train.py` — the only file you edit (model, Hopf modules, optimizer, training loop)
4. **Verify data exists**: Check `~/.cache/autoresearch/` for data shards and tokenizer. If missing, run `uv run prepare.py --num-shards 4` (small Mac runs) or `uv run prepare.py` (full).
5. **Initialize `results.tsv`**: header row only. Baseline recorded after first run.
6. **Confirm and go.**

## Experimentation

Each experiment runs on a single device (MPS on Apple Silicon, CUDA if available). Training uses a **fixed 5-minute wall-clock budget** (`TIME_BUDGET` in `prepare.py`).

Launch: `uv run train.py`

### What you CAN do

- Modify `train.py` only — architecture, Hopf hyperparameters, optimizer, batch size, depth, etc.
- Toggle `USE_HOPF_LATENT`, `HOPF_NUM_LEVELS`, `HOPF_MIX_ALPHA`, `HOPF_PLACEMENT`
- Add ablations: hypersphere projection, complex-phase-only, nested depth variants
- Tune Mac-friendly defaults (`DEPTH`, `DEVICE_BATCH_SIZE`, `TOTAL_BATCH_SIZE`, `WINDOW_PATTERN`)

### What you CANNOT do

- Modify `prepare.py` (fixed metric harness)
- Install new packages (only `pyproject.toml` deps)
- Change `evaluate_bpb` or `TIME_BUDGET`

### Goal

**Minimize `val_bpb`** (validation bits per byte). Lower is better. Fixed time budget makes experiments comparable.

**VRAM** is a soft constraint. Small increases OK for meaningful gains; avoid OOM.

**Simplicity criterion:** prefer simpler changes with equal or better `val_bpb`. Deleting code that doesn't help is a win.

### First run

Always establish the **baseline** first with `USE_HOPF_LATENT = False`. Record that `val_bpb` before testing Hopf variants.

## Hopf-specific experiment queue

After baseline, try these in order (skip/adapt if they crash):

| Priority | Experiment | Key change |
|----------|------------|------------|
| 1 | `hopf-input-l2` | `USE_HOPF_LATENT=True`, `HOPF_NUM_LEVELS=2`, `HOPF_PLACEMENT="input"` |
| 2 | `hopf-input-l3` | `HOPF_NUM_LEVELS=3` |
| 3 | `hopf-alpha-sweep` | `HOPF_MIX_ALPHA` in {0.25, 0.5, 0.75} |
| 4 | `hopf-per-block` | `HOPF_PLACEMENT="per_block"` |
| 5 | `hypersphere-ablation` | `USE_HOPF_LATENT=False`, `USE_HYPERSPHERE_LATENT=True` |
| 6 | `hopf+smaller-depth` | Hopf on with `DEPTH=4` vs baseline `DEPTH=4` param-matched |
| 7 | `hopf-phase-loss` | optional auxiliary phase-consistency regularizer (small weight) |

Document each run in `results.tsv` with a `hopf-*` prefix in the description.

## Output format

```
---
val_bpb:          0.997900
training_seconds: 300.1
...
```

Extract: `grep "^val_bpb:\|^peak_vram_mb:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	val_bpb	memory_gb	status	description
```

Status: `keep`, `discard`, or `crash`. Do not commit `results.tsv`.

## Experiment loop

Dedicated branch `autoresearch/<tag>`.

LOOP FOREVER:

1. Check git state
2. Edit `train.py` with one experimental idea
3. `git commit`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. On crash: `tail -n 50 run.log`, fix or discard
7. Log to `results.tsv`
8. If `val_bpb` improved → keep commit; else `git reset`

**Timeout:** >10 min → kill, discard, revert.

**NEVER STOP** once the loop begins. Run autonomously until interrupted.

## Success criteria for the research program

- Hopf variant beats baseline `val_bpb` by ≥0.5% relative, OR
- Matches baseline with fewer params / lower VRAM, OR
- Shows consistent improvement across ≥3 independent run tags

If Hopf consistently loses to hypersphere and baseline, document negative result — that is valuable science.