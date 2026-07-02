# Contributing to Hopf Autoresearch

Thank you for your interest in contributing. This project is MSAI thesis research at Northeastern University, and we actively welcome collaborators — researchers, students, and practitioners who want to push geometric inductive biases forward.

## Ways to contribute

You do not need permission to fix typos or docs. For larger work, open an issue first so we avoid duplicate effort.

### Research contributions (highest value)

1. **Reproduce our results** on your hardware (GPU preferred) and report in an issue
2. **Add eval benchmarks** where Hopf geometry should matter (phase, cycles, modular arithmetic)
3. **Run ablations** we have not tried (document in `results.tsv` format)
4. **Improve the paper** — clarity, figures, related work (PRs to `paper/`)

### Engineering contributions

1. **Platform support** — CUDA/Linux fork, Windows, CPU-only smoke tests
2. **CI** — GitHub Actions that run `structured_tasks.py` on PRs
3. **Refactor** `hopf_layers.py` without changing experiment outcomes
4. **Notebooks** — `analysis.ipynb` extensions, result visualization

## Development setup

```bash
git clone https://github.com/jchacker5/hopf-autoresearch.git
cd hopf-autoresearch
uv sync
uv run prepare.py --num-shards 4
uv run python -m py_compile train.py hopf_layers.py
```

## Branch naming

| Prefix | Use |
|--------|-----|
| `feat/` | New feature or experiment harness |
| `fix/` | Bug fix |
| `exp/` | Experiment branch (autoresearch style) |
| `docs/` | Documentation only |
| `paper/` | Paper / figure updates |

## Pull request guidelines

1. **One logical change per PR** — don't mix paper edits with `train.py` architecture changes
2. **Describe motivation** — which hypothesis (H1–H5) does this test?
3. **Include numbers** — if you ran training, paste `val_bpb` or task metrics
4. **Keep `prepare.py` stable** unless you have strong reason to change the eval harness (discuss in issue first)
5. **Do not commit** `results.tsv`, `run.log`, `.venv/`, or downloaded data

## Experiment logging format

If you run autoresearch-style experiments, log to a local `results.tsv`:

```
commit	val_bpb	memory_gb	status	description
```

Status: `keep`, `discard`, or `crash`.

## Code of conduct

Be respectful and scientific. Negative results are valuable — report them prominently.

## Questions?

Open a [GitHub Issue](https://github.com/jchacker5/hopf-autoresearch/issues) or email [defendre.j@northeastern.edu](mailto:defendre.j@northeastern.edu).