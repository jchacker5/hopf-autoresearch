#!/usr/bin/env python3
"""Multi-seed LM comparison for publication: baseline vs Hopf winner vs hypersphere."""

import json
import re
import subprocess
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "train.py"
OUT = ROOT / "experiments" / "results_lm_multiseed.json"

CONFIGS = [
    dict(name="baseline", hopf=False, hypersphere=False, levels=2, alpha=0.05, placement="per_block"),
    dict(name="hopf_winner", hopf=True, hypersphere=False, levels=2, alpha=0.05, placement="per_block"),
    dict(name="hypersphere", hopf=False, hypersphere=True, levels=2, alpha=0.05, placement="per_block"),
    dict(name="hopf_input", hopf=True, hypersphere=False, levels=2, alpha=0.05, placement="input"),
    dict(name="hopf_alpha_0.08", hopf=True, hypersphere=False, levels=2, alpha=0.08, placement="per_block"),
    dict(name="hopf_alpha_0.03", hopf=True, hypersphere=False, levels=2, alpha=0.03, placement="per_block"),
]
SEEDS = [42, 43]  # extend to [42, 43, 44] for final thesis tables


def patch(cfg, seed):
    text = TRAIN.read_text()
    text = re.sub(r"USE_HOPF_LATENT = \w+", f"USE_HOPF_LATENT = {cfg['hopf']}", text)
    text = re.sub(r"USE_HYPERSPHERE_LATENT = \w+", f"USE_HYPERSPHERE_LATENT = {cfg['hypersphere']}", text)
    text = re.sub(r"HOPF_NUM_LEVELS = \d+", f"HOPF_NUM_LEVELS = {cfg['levels']}", text)
    text = re.sub(r"HOPF_MIX_ALPHA = [\d.]+", f"HOPF_MIX_ALPHA = {cfg['alpha']}", text)
    text = re.sub(r'HOPF_PLACEMENT = "[^"]+"', f'HOPF_PLACEMENT = "{cfg["placement"]}"', text)
    text = re.sub(r"torch\.manual_seed\(\d+\)", f"torch.manual_seed({seed})", text)
    if "torch.cuda.manual_seed" in text:
        text = re.sub(r"torch\.cuda\.manual_seed\(\d+\)", f"torch.cuda.manual_seed({seed})", text)
    TRAIN.write_text(text)


def run_train():
    r = subprocess.run("uv run train.py > run.log 2>&1", shell=True, cwd=ROOT)
    log = (ROOT / "run.log").read_text()
    for line in log.splitlines():
        if line.startswith("val_bpb:"):
            return float(line.split()[1]), r.returncode == 0
    return None, False


def main():
    runs = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            print(f"Running {cfg['name']} seed={seed}...", flush=True)
            patch(cfg, seed)
            bpb, ok = run_train()
            runs.append({**cfg, "seed": seed, "val_bpb": bpb, "success": ok})
            print(f"  val_bpb={bpb}", flush=True)

    summary = {}
    for cfg in CONFIGS:
        bpbs = [r["val_bpb"] for r in runs if r["name"] == cfg["name"] and r["val_bpb"] is not None]
        if bpbs:
            summary[cfg["name"]] = {
                "mean": statistics.mean(bpbs),
                "std": statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0,
                "n": len(bpbs),
            }

    OUT.write_text(json.dumps({"runs": runs, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()