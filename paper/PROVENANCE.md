# Provenance: paper table / figure → command

For every table and figure: the command that produces its CSVs, where they
land, the command that renders it, and the observed runtime. Every render
reads CSVs only (no GPU). Run all commands from the repository root.

Each entry names its **protocol**:

- **per-layer**: every analysed layer is pruned in isolation, with
  calibration statistics from the dense model.
- **whole-model**: the official Wanda/SparseGPT code prunes the whole model
  sequentially.

Table numbers given in REPRODUCIBILITY_TASK.md are used. The other tables
are identified by their LaTeX `\label`. TODO (authors): fill in the
camera-ready numbers.

Runtimes are wall-clock times read from the original result-file timestamps
(one RTX A6000 48 GB), unless marked as measured in this release.

---

### Table 1: transformations evaluated (`tab:transforms`)
- data: none (static; lists the 13 transforms in `src/pruning_transforms/transforms.py`)
- render: `python analysis/generate_tables.py --table transforms`

### Table 2: main Δ across model scales (`tab:delta`). **Per-layer protocol**
- data:
  ```
  python experiments/run_experiments.py --models opt-125m opt-1.3b opt-6.7b llama-7b llama-13b
  ```
- file: `results/perlayer/{model}/transform_results.csv`
- render: `python analysis/generate_tables.py --table delta --models opt-125m opt-1.3b opt-6.7b llama-7b llama-13b`
- runtime: OPT-125M: RUNTIME_PERLAYER_OPT125M (measured in this release, smoke test); larger models take hours (CPU-side LU/PCA dominate)
- note: the per-layer LLaMA-7B/13B main-sweep CSVs have no `Hadamard_right`
  rows. The original script crashed on Hadamard (README §Corrections 5), so
  those cells render as `[X]`. LLaMA-7B Hadamard was later computed in the
  rebuttal run (`results/perlayer/llama-7b/sp50/`, Δ = 0.0). LLaMA-13B
  Hadamard was never computed and is left empty.

### Table 2b: dense zero rows of QR / SVD two-sided (`tab:qrsvd`). **Per-layer protocol**
- data/file: same as Table 2
- render: `python analysis/generate_tables.py --table qrsvd --models ...`

### Table 3: generality across pruning methods. **Whole-model protocol**
- data:
  ```
  python experiments/run_wholemodel_experiments.py --models opt-125m \
      --methods wanda sparsegpt --sparsity 0.5 --seeds 0 1 2
  python experiments/run_wholemodel_experiments.py --models opt-125m \
      --methods magnitude_perrow --sparsity 0.5 --seeds 0      # deterministic, one seed
  ```
- file: `results/wholemodel/opt-125m/{method}/sp50/seed{0,1,2}/delta_per_transform.csv`
- render: `python analysis/generate_tables.py --table methods --results_dir results/wholemodel --model opt-125m`
  (`--results_dir results/sample/wholemodel` renders the committed sample)
- runtime: ≈ 18–20 min per (method, seed) for OPT-125M

### Table 4: rank / spectral / mask-correlation diagnostics across scales (`tab:diagnostics`). **Per-layer protocol**
- data/file: as Table 2, `results/perlayer/{model}/layer_diagnostics.csv`
- render: `python analysis/generate_tables.py --table diagnostics --models ...`

### Column-sparsity table: Δ(zero_cols) per transform × method. **Whole-model protocol**
- data/file: as Table 3
- render: `python analysis/generate_tables.py --table colsparsity --results_dir results/wholemodel --model opt-125m`

### Calibration-seed variance (E1). **Whole-model protocol**
- data: as Table 3 (seeds 0 1 2). The script also writes `results/wholemodel/seed_variance_summary.csv`.
- render: `python analysis/generate_tables.py --table seeds --results_dir results/wholemodel`

### out_proj anomaly table (`tab:outproj`). **Per-layer protocol**
- data/file: as Table 2. It reports the `out_proj`/`o_proj` rows with rank drop > 10.
- render: `python analysis/generate_tables.py --table outproj --models ...`

### out_proj anomaly mechanism (E4). **Whole-model protocol**
- data:
  ```
  python experiments/run_oproj_mechanism.py --models opt-125m --methods wanda sparsegpt --sparsity 0.5
  python experiments/run_oproj_mechanism.py --models llama-7b --methods wanda --sparsity 0.5 --dtype float16
  ```
- file: `results/oproj_mechanism/{model}/{method}/sp50/{layer_mechanism,per_head_detail,summary_by_layer_type,target_depth_trend}.csv`
- render: `python analysis/generate_tables.py --table oproj_mechanism`
- supporting check (dead heads, prints only): `python experiments/check_dead_heads.py`

### Row-deletion result (E3) and perplexity check. **Whole-model protocol**
- data:
  ```
  python experiments/run_wholemodel_experiments.py --models llama-7b --methods wanda \
      --sparsity 0.5 --seeds 0 --truncate_oproj 1 --eval_ppl
  ```
- file: `results/wholemodel/llama-7b/wanda/sp50/seed0/run_info.json`
  (`ppl_wikitext2_pruned` = 7.2568, `ppl_wikitext2_truncated` = 7.2570,
  `truncate_oproj.rows_deleted` = 618)
- render: `python analysis/generate_tables.py --table ppl --results_dir results/wholemodel`
- runtime: not recorded (the original run left only end-of-run timestamps)

### Per-row magnitude out_proj control. **Whole-model protocol**
- data/file: Table 3's `magnitude_perrow` run, `transform_results.csv` + `layer_diagnostics.csv`
- render: `python analysis/generate_tables.py --table outproj_perrow --results_dir results/wholemodel`
  (this was `rock.py` in the research directory. Its Δ line printed `nan`, see README.)

### Cross-layer rotation table (E2). **Whole-model protocol**
- data:
  ```
  python experiments/run_crosslayer_q.py --models opt-125m --methods wanda sparsegpt \
      --sparsity 0.5 --seed 0 --q_modes global per_block random
  ```
- file: `results/crosslayer_q/opt-125m/{method}/sp50/{run_info.json,delta_{global,per_block,random}.csv}`
- render: `python analysis/generate_tables.py --table crosslayer`

### Synthetic three-mechanism factorial (E5). CPU only, no model
- data: `python experiments/synthetic_mechanism_ablation.py`
- file: `results/synthetic/synthetic_mechanism_ablation_summary.csv` (and `_raw.csv`, `e5_part_*.csv`)
- render: `python analysis/generate_tables.py --table synthetic`
  (`--results_dir results/sample/synthetic` renders the committed sample)
- note: the original outputs were never saved. The committed sample was
  produced by this release's code on 2026-10-18.

### Rebuttal tables: sparsity ablation, Wanda vs SparseGPT, out_proj vs SparseGPT. **Per-layer protocol**
- data:
  ```
  python experiments/run_experiments_rebuttal.py --models opt-1.3b --sparsity 0.5 0.6 0.7 0.8
  python experiments/run_experiments_rebuttal.py --models opt-125m llama-7b --sparsity 0.5
  ```
- file: `results/perlayer/{model}/sp{50,60,70,80}/transform_results.csv`, `layer_diagnostics.csv`
- render: `python analysis/generate_tables.py --table rebuttal --models opt-125m opt-1.3b`
- runtime: ≈ 3 h 50 min per sparsity level for OPT-1.3B with SparseGPT
- note: `rebuttal_tables.tex` in the research directory was generated on
  2026-06-19, *before* the 2026-06-22..24 reruns that produced these CSVs.
  It is stale. Regenerate rather than copy it.

### Pre-pruning rotation table (`tab:rotation`). **Not reproducible**
- data: produced by `test_hypothesis.py` / `test_hypothesis_v2.py`, which are
  not in this repository and were not found anywhere on the authors' machine.
  The generator prints the hard-coded values with a warning comment.
- render: `python analysis/generate_tables.py --table rotation`

### Appendix A1: all metrics × thresholds (`tab:app_metrics`). **Per-layer protocol**
- data/file: as Table 2, plus `results/perlayer/{model}/sensitivity.csv`
- render: `python analysis/generate_tables.py --table appendix --models ...`
- note: see README "Known gaps" (the threshold rows compare different
  thresholds for the pruned and dense sides). Reproduced as published,
  pending the authors' decision.

### Figures 1–5 (Section 5). **Per-layer protocol**
- data/file: as Table 2
- render: `python analysis/generate_figures.py --models opt-125m opt-1.3b opt-6.7b llama-7b llama-13b`
  - Fig. 1 `figures/fig1_rank_drop.png`: rank drop across thresholds
  - **Fig. 2 `figures/fig2_spectral.png`: spectral figure** (R10 vs normalised depth)
  - Fig. 3 `figures/fig3_mask_corr.png`: ρ(M) Wanda vs random vs magnitude
  - Fig. 4 `figures/fig4_diagnostic.png`: rank drop / R10 vs max Δ
  - Fig. 5 `figures/fig5_heatmap.png`: mean Δ per transform × layer type

### Everything at once
```
python analysis/generate_tables.py --table all --out paper/tables.tex
```
