# Scattered by Design

Code for **Scattered by Design: Why Per-Output Pruning Resists Structured
Compression** (NeurIPS 2026).

Per-output pruning methods (Wanda, SparseGPT, per-row magnitude) produce
unstructured sparsity, and no post-hoc linear transformation turns it into
structured sparsity. We measure

    Δ(T, W) = zero_rows(T(W_pruned)) − zero_rows(T(W_dense))

over 13 transformations (rotations, decompositions, statistical rescalings,
permutations), three pruning methods and OPT/LLaMA models. Δ ≤ 0.4%
throughout. Three mechanisms explain this:

1. pruning preserves rank;
2. pruning disperses rather than concentrates spectral energy;
3. masks of different rows are almost uncorrelated.

The one exception, early-block attention output projections, comes from dead
attention heads.

Pruning is delegated to the **official Wanda and SparseGPT implementations**
(`locuslab/wanda`, vendored in `third_party/wanda`). Whole-model Wanda on
LLaMA-7B at 50% sparsity reproduces the published WikiText-2 perplexity:
**7.257 vs 7.26** in Sun et al.

---

## Quickstart (one headline number, < 20 min)

```bash
pip install -r requirements.txt     # pinned; see Setup

# 1. No GPU, seconds: render Table 3 (pruning methods) from the committed sample
python analysis/generate_tables.py --table table3 --results_root results/sample

# 2. One GPU, ~5 min: recompute the OPT-125M whole-model Wanda result from scratch
python experiments/run_wholemodel_experiments.py --models opt-125m --methods wanda --sparsity 0.5 --seeds 0
cat results/wholemodel/opt-125m/wanda/sp50/seed0/delta_per_transform.csv
```

The largest Δ(zero_rows) is **+0.0031 (0.31%), at SVD-left** (and SVD-weighted).
LU-left gives +0.00017 and QR-left +0.00015. All other transforms give
exactly 0. `bash scripts/smoke_test.sh` checks these values, plus the
synthetic ablation and the per-layer pipeline, against fixed tolerances.

## Setup

```bash
conda create -n scattered-by-design python=3.9.13 && conda activate scattered-by-design
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

- The Wanda/SparseGPT dependency is **vendored** in `third_party/wanda`, with
  the needed patches already applied. There is no clone or patch step.
  Every script that uses it defaults to `--wanda_repo third_party/wanda`.
- Scripts add `src/` to `sys.path` themselves. `pip install -e .` also works.
- Models (`facebook/opt-*`, `huggyllama/llama-*`) and C4 / WikiText-2 are
  downloaded from the Hugging Face Hub on first use.

### Environment that produced the published numbers

`requirements.txt` pins the exact versions in the authors' `kita` env:
Python 3.9.13, torch 2.8.0+cu128 (CUDA 12.8), transformers 4.57.1,
datasets 3.5.0, numpy 2.0.2, scipy 1.13.1, scikit-learn 1.6.1,
pandas 2.2.3. Do not upgrade any of them to silence warnings.

The two patches to the vendored Wanda tree (`third_party/wanda/MODIFICATIONS.md`)
are needed only *because* these versions are newer than the 2023 code expects:

- `datasets` dropped the `'allenai--c4'` config;
- `transformers` now passes LLaMA rotary embeddings into decoder layers.

With older libraries you would not need them. Both patches are
backward-compatible: the rotary fix forwards `position_embeddings` only when
the kwarg exists. The vendored tree therefore runs under both old and new
`transformers`.

Hardware: NVIDIA RTX A6000 (48 GB), driver 580.65.06.

```
# nvidia-smi -i 4 (authors' node, 2026-10-18, during the release smoke test;
# the node has several RTX A6000s and is shared, so memory is partly used by other jobs)
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
# +-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |=========================================+========================+======================|
# |   4  NVIDIA RTX A6000               Off |   00000000:89:00.0 Off |                  Off |
# | 30%   57C    P2            112W /  300W |   34114MiB /  49140MiB |     23%      Default |
# +-----------------------------------------+------------------------+----------------------+
# torch 2.8.0+cu128: torch.version.cuda = 12.8
```

## Hardware and runtime

One RTX A6000 48 GB. Times marked (m) were measured for this release; the
others are read from the original result-file timestamps.

| Experiment | Script | Runtime |
|---|---|---|
| Smoke test (all three checks) | `scripts/smoke_test.sh` | ≈ 10 min (m) |
| Synthetic factorial (CPU only) | `experiments/synthetic_mechanism_ablation.py` | ≈ 1 min, 8 BLAS threads (m) |
| Per-layer main sweep, OPT-125M | `experiments/run_experiments.py` | ≈ 5.5 min (m) |
| Per-layer rebuttal, OPT-1.3B, one sparsity, with SparseGPT | `experiments/run_experiments_rebuttal.py` | ≈ 3 h 50 min |
| Whole-model, OPT-125M, Wanda, one seed | `experiments/run_wholemodel_experiments.py` | ≈ 4.5 min (m); SparseGPT and `--eval_ppl` add to this |
| Whole-model, LLaMA-7B (fp16) | `experiments/run_wholemodel_experiments.py` | fits in 48 GB; time not recorded |
| Cross-layer Q, OPT-125M, one method | `experiments/run_crosslayer_q.py` | ≈ 2–3 min |
| out_proj mechanism, OPT-125M, one method | `experiments/run_oproj_mechanism.py` | ≈ 2 min |
| Table / figure rendering | `analysis/*.py` | seconds, no GPU |

LLaMA-13B and OPT-13B need a larger GPU or several GPUs (the vendored Wanda
code handles `hf_device_map`).

## Directory map

```
README.md  LICENSE  requirements.txt  environment.yml  pyproject.toml
third_party/wanda/       vendored locuslab/wanda@8e8fc87 (MIT), patched; NOTICE.md, MODIFICATIONS.md
patches/                 the two patches as diffs (documentation, not a build step)
src/pruning_transforms/  single source of truth, imported by every experiment
  transforms.py          the 13 transforms (TRANSFORMS) + Identity baseline (BASELINE)
  metrics.py             zero_rows/cols, near_zero, block4x4, rank, spectral, mask_corr
  pruning.py             per-layer pruners, Wanda scaler, per-layer SparseGPT, official-code import
  config.py              thresholds and constants (do not change), model registry, 10-bin sampling
experiments/
  run_experiments.py             per-layer protocol, main sweep (Tables 2, 4, 6, 7, A1; Fig. 2)
  run_experiments_rebuttal.py    per-layer protocol + per-layer SparseGPT + 50-80% sparsity sweep
  run_wholemodel_experiments.py  whole-model protocol (official code; Tables 3, A3); --seeds, --truncate_oproj, --eval_ppl
  run_crosslayer_q.py            SliceGPT-style learned cross-layer rotation (Table 5)
  run_oproj_mechanism.py         per-head analysis of the out_proj anomaly, with controls (Tables A7, A8)
  check_dead_heads.py            dead-head count behind the out_proj anomaly (prints only)
  synthetic_mechanism_ablation.py  synthetic three-mechanism factorial, CPU only (Table A6)
analysis/
  generate_tables.py     every LaTeX table, from CSVs only (--table camera_ready: Tables 1-7, A1-A8)
  generate_figures.py    Fig. 2 (R10 vs depth) plus exploratory figures, from per-layer CSVs
scripts/smoke_test.sh    end-to-end check of both protocols
paper/PROVENANCE.md      camera-ready table/figure -> command map
paper/*.pdf              the camera-ready paper
results/                 outputs (gitignored), except results/sample/ (committed)
tests/                   unit tests of the shared library
```

## Output schemas

All Δ values are pruned − dense, as fractions (0.0031 = 0.31%).

**Per-layer protocol**: `results/perlayer/<model>/` (main sweep) or
`results/perlayer/<model>/sp<NN>/` (rebuttal sweep):

| File | One row per | Columns |
|---|---|---|
| `transform_results.csv` | layer × transform × pruning | `model, layer, layer_type, [sparsity,] C_out, C_in, block_idx, depth_bin, norm_depth, transform, pruning ∈ {wanda, magnitude_global, dense[, sparsegpt]}, zero_rows, zero_cols, near_zero_rows, block4x4` |
| `layer_diagnostics.csv` | layer | `{dense,wanda,mag,mag_pr,hi90[,sgpt]}_rank_1em{3,4,5,6}` (rank at t·σ_max(dense)); `drop_{wanda,mag,hi90[,sgpt]}_rank_1em*` (dense − pruned; **negative values are real**); `eff_rank_*`; `{wanda,mag,hi90[,sgpt]}_top{5,10,20,50}_ratio`; `mask_corr_{wanda,mag_g,mag_pr,random[,sgpt]}`; `qr_theory_zero_rows` |
| `sensitivity.csv` | layer × transform × threshold (Wanda[, SparseGPT]) | `model, layer, [pruning,] transform, threshold ∈ {1e-3,1e-4,1e-6}, zero_rows` |
| `transform_failures.csv` | failed (transform, layer, pruning) | `model, layer, sparsity, transform, pruning, C_out, C_in, error_type, error_msg`. Must be empty. `transform="(pruning)"` marks a failed SparseGPT prune. |
| `summary.json` | run | averages, `delta_per_transform`, `max_delta_wanda`, `n_transform_failures` |
| `all_models_summary.csv` / `rebuttal_summary.csv` | model [× sparsity] | cross-model summary |

**Whole-model protocol**: `results/wholemodel/<model>/<method>/sp<NN>/seed<S>/`:

| File | One row per | Columns |
|---|---|---|
| `transform_results.csv` | layer × transform × {method, dense} | `model, layer, method, sparsity, seed, transform, pruning, zero_rows, zero_cols, near_zero_rows, block4x4` |
| `layer_diagnostics.csv` | layer | `dense_rank_1em*, pruned_rank_1em*, drop_rank_1em*, eff_rank_{dense,pruned}, top{5,10,20,50}_ratio, mask_corr, actual_sparsity` |
| `delta_per_transform.csv` | transform (14 rows) | `transform, zero_rows, zero_cols, block4x4` (mean Δ over sampled layers) |
| `transform_failures.csv` | failure | must be empty |
| `run_info.json` | run | achieved sparsity, `ppl_wikitext2_pruned`, `max_delta_zero_rows`, `best_transform`, `truncate_oproj{...}`, `ppl_wikitext2_truncated` |
| `../../../../seed_variance_summary.csv` | model × method × sparsity × transform × metric | `n_seeds, delta_mean, delta_std, t_stat, p_value` |

**Cross-layer Q** (`results/crosslayer_q/<model>/<method>/sp<NN>/`):
`crosslayer_q_results.csv` (layer × q_mode × pruning: `zero_rows, zero_cols,
near_zero_rows, near_zero_cols, block4x4`), `delta_<q_mode>.csv` (per-layer
Δ), `run_info.json` (`delta_summary`).

**out_proj mechanism** (`results/oproj_mechanism/<model>/<method>/sp<NN>/`):
`layer_mechanism.csv` (layer: `xn_across_head_share, xn_head_norm_cv, rho_M,
spearman_headnorm_zerorate, head_zero_rate_{max,min}, frac_dead_cols,
frac_heads_mostly_dead, rank_drop_1em{3,6}, R10, delta_svdleft_zero_rows`),
`per_head_detail.csv`, `summary_by_layer_type.csv`, `target_depth_trend.csv`.

**Synthetic** (`results/synthetic/`): `synthetic_mechanism_ablation_raw.csv`
(spectrum × rank × mask × seed: `rho_M, rank_dense, rank_pruned, rank_drop,
R10, delta_<T>_<metric>`) and `synthetic_mechanism_ablation_summary.csv`
(mean over seeds).

## Two pruning protocols

Both appear in the paper, distinguished in the table captions. They are
deliberately kept separate.

- **Per-layer** (`run_experiments.py`, `run_experiments_rebuttal.py`):
  - Each analysed layer is pruned in isolation.
  - Calibration statistics (Wanda scalers; the SparseGPT Hessian) come from
    one forward pass of the **dense** model: 128 C4 sequences of 128 tokens.
  - This produced the main sweep across model scales (Tables 2, 4, 6 and 7).
- **Whole-model** (`run_wholemodel_experiments.py`, also `run_crosslayer_q.py`
  and `run_oproj_mechanism.py`):
  - Pruning is delegated to the official `locuslab/wanda` code.
  - Pruning is sequential: block *i* is pruned on activations propagated
    through the already-pruned blocks 0..*i*−1 (128 samples × 2048 tokens).
  - This produced the OPT-125M / LLaMA-7B verification runs, the
    calibration-variance study and the row-deletion experiment.

Both protocols analyse the same layers. They split model depth into 10 equal
bins and take every linear layer type in the central block of each bin. Rank
thresholds are always anchored to σ_max of the *dense* matrix.

## Vendored code: attribution

`third_party/wanda/` is a copy of [locuslab/wanda](https://github.com/locuslab/wanda)
at commit `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`. Its authors are
Mingjie Sun, Zhuang Liu, Anna Bair and J. Zico Kolter. It is redistributed
under its MIT licence (`third_party/wanda/LICENSE`, retained verbatim) with
two local modifications. See `third_party/wanda/NOTICE.md` and
`MODIFICATIONS.md`. Its SparseGPT path derives from IST-DASLab/sparsegpt
(Frantar & Alistarh, 2023).

## Citation

```bibtex
@inproceedings{omolaoye2026scattered,
  title     = {Scattered by Design: Why Per-Output Pruning Resists Structured Compression},
  author    = {Omolaoye, Victor A. and de Melo, Gerard},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=P0XAEBRFU2}
}
```

If you use the vendored pruning code, please also cite Wanda:

```bibtex
@article{sun2023wanda,
  title   = {A Simple and Effective Pruning Approach for Large Language Models},
  author  = {Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year    = {2023},
  journal = {arXiv preprint arXiv:2306.11695}
}
```
