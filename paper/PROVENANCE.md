# Provenance: camera-ready table / figure → command

For every table and figure in the camera-ready paper
(`paper/Scattered_by_Design__Why_Per_Output_Pruning_Resists_Structured_Compression___camera_ready.pdf`):
the command that produces its CSVs, where they land, and the command that
renders it. Rendering reads CSVs only (no GPU). Run everything from the
repository root.

Each entry names its **protocol**:

- **per-layer**: every analysed layer is pruned in isolation, with
  calibration statistics from the dense model (`experiments/run_experiments*.py`).
- **whole-model**: the official Wanda/SparseGPT code prunes the whole model
  sequentially (`experiments/run_wholemodel_experiments.py` and the scripts
  built on it).

The experiment scripts write under one results root:

```
results/perlayer/{model}/                         per-layer main sweep
results/perlayer/{model}/sp{50,60,70,80}/         per-layer rebuttal sweep
results/wholemodel/{model}/{method}/sp50/seed{S}/ whole-model runs
results/crosslayer_q/{model}/{method}/sp50/       learned cross-layer rotations
results/oproj_mechanism/{model}/{method}/sp50/    out_proj anomaly mechanism
results/synthetic/                                synthetic factorial
```

Render every camera-ready table at once (tables whose inputs are missing are
skipped with a message):

```
python analysis/generate_tables.py --table camera_ready --out paper/tables.tex
python analysis/generate_tables.py --table camera_ready --results_root results/sample   # committed sample only
```

Each table can also be rendered on its own with `--table table1` … `table7`,
`a1` … `a8`. All of them were rendered from the original research CSVs on
2026-10-18 and compared cell by cell with the camera-ready PDF. They match
except where an entry below says otherwise.

Runtimes are wall-clock on one RTX A6000 48 GB. "Measured" means measured in
this release; the others are read from the original result-file timestamps.

---

## Main paper

### Figure 1: transformation schematic
- Conceptual illustration, no data.

### Table 1: transformations evaluated (`tab:transforms`)
- data: none (lists the 13 transforms in `src/pruning_transforms/transforms.py`
  and the three cross-layer variants in `experiments/run_crosslayer_q.py`)
- render: `python analysis/generate_tables.py --table table1`

### Table 2: Δ across model scales (`tab:delta`). **Per-layer**, whole-model in brackets
- data:
  ```
  python experiments/run_experiments.py --models opt-125m opt-1.3b opt-6.7b llama-7b llama-13b
  python experiments/run_experiments_rebuttal.py --models llama-7b --sparsity 0.5     # Hadamard cell, see note
  python experiments/run_wholemodel_experiments.py --models opt-125m --methods wanda --sparsity 0.5 --seeds 0 1 2
  python experiments/run_wholemodel_experiments.py --models llama-7b --methods wanda --sparsity 0.5 --seeds 0 \
      --truncate_oproj 1 --eval_ppl
  ```
- file: `results/perlayer/{model}/transform_results.csv`;
  brackets: `results/wholemodel/{opt-125m,llama-7b}/wanda/sp50/seed*/delta_per_transform.csv`
- render: `python analysis/generate_tables.py --table table2`
- runtime: per-layer OPT-125M ≈ 5.5 min (measured); larger models take hours
  (CPU-side LU/PCA dominate). Whole-model OPT-125M ≈ 4.5 min per seed (measured).
- note: the main-sweep LLaMA CSVs have no `Hadamard_right` rows. The
  generator takes the LLaMA-7B cell from `results/perlayer/llama-7b/sp50/`
  (0.0%) and marks this in a LaTeX comment. No LLaMA-13B Hadamard result
  exists, so that cell renders `[X]` (the PDF prints 0.0%).

### Table 3: generality across pruning methods (`tab:methods`). **Whole-model**
- data:
  ```
  python experiments/run_wholemodel_experiments.py --models opt-125m \
      --methods wanda sparsegpt --sparsity 0.5 --seeds 0 1 2
  python experiments/run_wholemodel_experiments.py --models opt-125m \
      --methods magnitude_perrow --sparsity 0.5 --seeds 0      # deterministic, one seed
  ```
- file: `results/wholemodel/opt-125m/{method}/sp50/seed{0,1,2}/delta_per_transform.csv`
- render: `python analysis/generate_tables.py --table table3`
- runtime: ≈ 4.5 min per (method, seed) for Wanda (measured); SparseGPT is slower

### Table 4: dense zero rows of QR / SVD two-sided (`tab:qrsvd`). **Per-layer**
- data/file: as Table 2
- render: `python analysis/generate_tables.py --table table4`

### Table 5: learned cross-layer rotations (`tab:crosslayer`). **Whole-model**
- data:
  ```
  python experiments/run_crosslayer_q.py --models opt-125m --methods wanda sparsegpt \
      --sparsity 0.5 --seed 0 --q_modes global per_block random
  ```
- file: `results/crosslayer_q/opt-125m/{method}/sp50/run_info.json`
- render: `python analysis/generate_tables.py --table table5`
- also: the 40.8% dead-column figure in Section 7 is the `near_zero_cols`
  max in the same file (`python analysis/generate_tables.py --table crosslayer`
  prints means with maxima in parentheses).

### Table 6: out_proj/o_proj anomaly layers (`tab:outproj`). **Per-layer**, whole-model in brackets
- data: as Table 2 (per-layer), plus
  ```
  python experiments/run_oproj_mechanism.py --models opt-125m --methods wanda sparsegpt --sparsity 0.5
  python experiments/run_oproj_mechanism.py --models llama-7b --methods wanda --sparsity 0.5 --dtype float16
  ```
- file: `results/perlayer/{model}/{layer_diagnostics,transform_results}.csv`;
  brackets: `results/oproj_mechanism/{model}/wanda/sp50/layer_mechanism.csv` (seed 0)
- render: `python analysis/generate_tables.py --table table6`

### Table 7: diagnostic metrics (`tab:diagnostics`). **Per-layer**
- data/file: as Table 2, `results/perlayer/{model}/layer_diagnostics.csv`
- render: `python analysis/generate_tables.py --table table7`
- also: the random-mask ρ(M) range 0.011–0.026 quoted in Sections 5.3/6.3 is
  the per-model mean of `mask_corr_random` in the same files.

### Figure 2: R10 across normalised depth. **Per-layer**
- data/file: as Table 2
- render: `python analysis/generate_figures.py --models opt-125m opt-1.3b opt-6.7b llama-7b llama-13b`
  → `figures/fig2_spectral.png` (regenerated image is identical to the PDF's).
  The script also writes four exploratory figures that are not in the paper.

### In-text numbers
- **LLaMA-7B perplexity 7.257 / 7.2568 → 7.2570, 618 rows deleted, rank 3478**
  (Sections 4.1, 5.2). Whole-model, from Table 2's LLaMA-7B command.
  File `results/wholemodel/llama-7b/wanda/sp50/seed0/run_info.json`. Render:
  `python analysis/generate_tables.py --table ppl`. The 28.5M vs 16.8M
  storage figure is 3478·(4096+4096) vs 4096².
- **Seed std ≤ 1.3×10⁻⁴** (Table 2 caption). From
  `results/wholemodel/seed_variance_summary.csv`, written by the whole-model
  script. Render: `python analysis/generate_tables.py --table seeds`.
- **SparseGPT out_proj rank-drop +12.0 → +0.5, Δ +1.5% → +0.07%** (Section 6.5).
  From `results/oproj_mechanism/opt-125m/{wanda,sparsegpt}/sp50/summary_by_layer_type.csv`.
  Render: `python analysis/generate_tables.py --table oproj_mechanism`.
- **Head-identity share up to 0.92 at out_proj; 23.6% dead columns in block 1**
  (Section 6.5, Appendix E). From
  `results/oproj_mechanism/llama-7b/wanda/sp50/layer_mechanism.csv`
  (`xn_across_head_share`, `frac_dead_cols`).
- **Sparsity sweep 50–80% on OPT-1.3B** (Section 4.1, Appendix F). **Per-layer**:
  ```
  python experiments/run_experiments_rebuttal.py --models opt-1.3b --sparsity 0.5 0.6 0.7 0.8
  ```
  File `results/perlayer/opt-1.3b/sp{50,60,70,80}/`. Render:
  `python analysis/generate_tables.py --table rebuttal --models opt-1.3b`.
  Runtime ≈ 3 h 50 min per sparsity level with SparseGPT.

## Appendix

### Table A1: all metrics × thresholds (`tab:app_metrics`). **Per-layer**
- data/file: as Table 2, plus `results/perlayer/{model}/sensitivity.csv`
- render: `python analysis/generate_tables.py --table a1`
- note: the Exact-rows rows at 10⁻³/10⁻⁴ compare pruned zero rows at that
  threshold with dense zero rows at 10⁻⁶ (only the 10⁻⁶ dense value is
  stored). The "fully zero" 4×4 blocks are blocks that are ≥ 90% zero.

### Table A2: pre-pruning rotation (`tab:rotation`). **Not reproducible from CSVs**
- data: produced by `test_hypothesis.py` / `test_hypothesis_v2.py`, which are
  not in this repository. The generator prints the published values with a
  warning comment.
- render: `python analysis/generate_tables.py --table a2`

### Table A3: complete per-transform Δ × method (`tab:app_methods`). **Whole-model**
- data/file: as Table 3
- render: `python analysis/generate_tables.py --table a3`

### Tables A4 and A5: out_proj diagnostics and Δ (`tab:app_outproj_diag`, `tab:app_outproj_delta`). **Per-layer**, whole-model in brackets
- data/file: as Table 6
- render: `python analysis/generate_tables.py --table a4` and `--table a5`
- note: † rows are layers that exceed rank-drop 10 at 10⁻³ only under
  whole-model pruning. Under that criterion the generator also lists
  LLaMA-7B `l.2.o_proj` (+14 / +0, Δ +0.0%), which the PDF omits.

### Table A6: synthetic factorial (`tab:synthetic`). CPU only, no model
- data: `python experiments/synthetic_mechanism_ablation.py`
- file: `results/synthetic/synthetic_mechanism_ablation_summary.csv` (and `_raw.csv`, `e5_part_*.csv`)
- render: `python analysis/generate_tables.py --table a6`
  (`--results_root results/sample` renders the committed sample)
- runtime: ≈ 1 min for the full factorial on CPU with 8 BLAS threads (measured)
- note: the original outputs were not kept. The committed sample comes from
  this release's code (2026-10-18).

### Table A7: layer-type comparison (`tab:app_layertype`). **Whole-model**
- data/file: as Table 6 (`run_oproj_mechanism.py`), `summary_by_layer_type.csv`
- render: `python analysis/generate_tables.py --table a7`

### Table A8: LLaMA-7B o_proj depth profile (`tab:app_depth`). **Whole-model**
- data/file: as Table 6, `results/oproj_mechanism/llama-7b/wanda/sp50/layer_mechanism.csv`
- render: `python analysis/generate_tables.py --table a8`
