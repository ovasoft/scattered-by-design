# results/

Experiment scripts write their outputs here, one subdirectory per protocol:
`perlayer/`, `wholemodel/`, `crosslayer_q/`, `oproj_mechanism/`,
`synthetic/`. Everything under `results/` is gitignored except this file
and `sample/`.

## `sample/` (committed, ~600 KB)

This sample lets the table generators run out of the box without a GPU:

| Path | Contents | Produced by |
|---|---|---|
| `sample/wholemodel/opt-125m/{wanda,sparsegpt}/sp50/seed{0,1,2}/` | `delta_per_transform.csv`, `layer_diagnostics.csv`, `run_info.json`, `transform_failures.csv` | original whole-model runs, 2026-07-24 |
| `sample/wholemodel/opt-125m/magnitude_perrow/sp50/seed0/` | the files above, plus `transform_results.csv` | original run, 2026-07-27 |
| `sample/wholemodel/llama-7b/wanda/sp50/seed0/` | the files above, including the perplexity and row-deletion record in `run_info.json` | original run, 2026-07-27 |
| `sample/wholemodel/seed_variance_summary.csv` | E1 seed-variance summary | original runs |
| `sample/synthetic/` | full synthetic factorial (4 cells × 5 seeds) | this release's code, 2026-10-18 (byte-identical to the original script's output; the original outputs were never saved) |

The whole-model `transform_results.csv` files (≈190 KB per run) are left out
of the sample, except for the per-row magnitude run. Rerun
`experiments/run_wholemodel_experiments.py` to regenerate them.

```bash
python analysis/generate_tables.py --table methods     --results_dir results/sample/wholemodel
python analysis/generate_tables.py --table colsparsity --results_dir results/sample/wholemodel
python analysis/generate_tables.py --table seeds       --results_dir results/sample/wholemodel
python analysis/generate_tables.py --table ppl         --results_dir results/sample/wholemodel
python analysis/generate_tables.py --table synthetic   --results_dir results/sample/synthetic
```
