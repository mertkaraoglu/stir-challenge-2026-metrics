# STIR 2026 — Metrics

Metric computation for the [STIR 2026](https://stirchallenge.org) point tracking benchmark. Reads inference outputs produced by [stir-challenge-2026-inference](https://github.com/mertkaraoglu/stir-challenge-2026-inference), computes AJ / ATA / OA, and generates comparison plots and a summary table.

## Requirements

- [uv](https://docs.astral.sh/uv/) package manager (no GPU required)

## Setup

```bash
uv sync
```

## Running

```bash
uv run run.py results/ --data_dir /path/to/dataset
```

`results/` is the output directory from the inference repo. `--data_dir` points to the original dataset (needed for ground-truth annotations).

## Outputs

```
results/
  mono/<model>/metrics.json         ← per-sequence + overall metrics
  stereo/<model>/metrics.json
  plots/
    mono_overall.png                ← AJ/ATA vs threshold for all mono models
    stereo_overall.png
    mono_<seq_id>.png               ← per-sequence comparison plots
    stereo_<seq_id>.png
    summary.md                      ← markdown table, also printed to stdout
```

## Metrics

| Metric | Definition |
|--------|-----------|
| **AJ** (Average Jaccard) | TP / (TP + FP + FN) at each threshold, averaged |
| **ATA** (Avg Tracking Accuracy) | Fraction of GT-visible points localised within threshold |
| **OA** (Occlusion Accuracy) | Visibility classification accuracy |
| **p95 latency** | 95th-percentile frame latency (ms), excluding first 10 frames warm-up |

Mono thresholds: `(2, 4, 8, 16, 32)` px  
Stereo thresholds: `(0.002, 0.004, 0.008, 0.016, 0.032)` m
