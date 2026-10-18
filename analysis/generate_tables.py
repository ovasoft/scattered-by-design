"""
Render the paper's LaTeX tables from result CSVs. Nothing is recomputed:
every number comes from a CSV / JSON written by an experiment script, so a
reader can check a number without a GPU.

Two result schemas exist (see README, "Output schemas"):

  perlayer    experiments/run_experiments*.py. Method-prefixed columns
              (drop_wanda_rank_1em3, mask_corr_wanda, wanda_top10_ratio, ...),
              one directory per model: <dir>/<model>/ (main sweep) or
              <dir>/<model>/sp<NN>/ (rebuttal sweep).
  wholemodel  experiments/run_wholemodel_experiments.py. A `method` column and
              generic names (drop_rank_1em3, mask_corr, top10_ratio), one
              directory per run: <dir>/<model>/<method>/sp<NN>/seed<S>/.

The schema is detected from the CSV header (override with --schema). Each
table states the schema it needs.

Usage:
  python analysis/generate_tables.py --table paper    --results_dir results/perlayer
  python analysis/generate_tables.py --table rebuttal --results_dir results/perlayer
  python analysis/generate_tables.py --table methods  --results_dir results/wholemodel
  python analysis/generate_tables.py --table all      # every table whose CSVs exist
  python analysis/generate_tables.py --table camera_ready   # camera-ready Tables 1-7, A1-A8
  python analysis/generate_tables.py --list
"""
import os, sys, json, glob, argparse
from collections import OrderedDict

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES  = os.path.join(REPO, "results")

PLACEHOLDER = r"\textbf{[X]}"

MNAME = {
    "opt-125m":"OPT-125M","opt-1.3b":"OPT-1.3B","opt-6.7b":"OPT-6.7B",
    "opt-13b":"OPT-13B","llama-7b":"LLaMA-7B","llama-13b":"LLaMA-13B",
}
METHOD_NAME = {"wanda": "Wanda", "sparsegpt": "SparseGPT",
               "magnitude_perrow": "Per-row magnitude", "magnitude": "Magnitude"}
TNAME = {
    "PCA_right"       : "PCA (right)",
    "SVD_right"       : r"SVD right ($V$)",
    "SVD_left"        : r"SVD left ($U^\top$)",
    "SVD_twosided"    : r"SVD two-sided ($\Sigma$)$^\dagger$",
    "QR_left"         : r"QR left ($\to R$)$^\dagger$",
    "LU_left"         : r"LU left ($\to U$)",
    "Hadamard_right"  : "Hadamard (right)",
    "DCT_right"       : "DCT (right)",
    "Diagonal_scaling": "Diagonal scaling",
    "Col_sort"        : "Column sort",
    "Row_sort"        : "Row sort",
    "SVD_weighted"    : "SVD weighted",
    "Cuthill_McKee"   : "Cuthill-McKee",
    "Identity"        : "Identity (baseline)",
}
TORDER = [
    "PCA_right","Hadamard_right","DCT_right",
    "SVD_right","SVD_left","SVD_twosided","QR_left","LU_left",
    "Diagonal_scaling","SVD_weighted",
    "Col_sort","Row_sort","Cuthill_McKee",
    "Identity",
]
TBREAKS = {"SVD_right","Diagonal_scaling","Col_sort","Identity"}
ALL_MODELS = ["opt-125m","opt-1.3b","opt-6.7b","opt-13b","llama-7b","llama-13b"]


# ── Schema detection ─────────────────────────────────────────────────────────
def detect_schema(results_dir):
    """'perlayer' or 'wholemodel', from the first transform_results.csv /
    layer_diagnostics.csv header found under results_dir."""
    for name in ("transform_results.csv", "layer_diagnostics.csv"):
        hits = sorted(glob.glob(os.path.join(results_dir, "**", name), recursive=True))
        if hits:
            cols = pd.read_csv(hits[0], nrows=0).columns
            if "method" in cols:
                return "wholemodel"
            if any(c.startswith(("drop_wanda_", "mask_corr_wanda")) for c in cols) \
                    or "pruning" in cols:
                return "perlayer"
    return None


def require(schema, needed, table):
    if schema != needed:
        sys.exit(f"--table {table} needs {needed} CSVs; {args.results_dir} "
                 f"looks like '{schema}'. Pass the right --results_dir (or --schema).")


def pct(v, nd=1):
    return f"{v*100:+.{nd}f}\\%"


# ══════════════════════════════════════════════════════════════════════════════
# PER-LAYER SCHEMA — main-paper tables (Tables 1, 2, 2b, diagnostics, out_proj,
# pre-pruning rotation, appendix A1). Ported from generate_tables2.py; output is
# byte-identical except where marked CHANGED.
# ══════════════════════════════════════════════════════════════════════════════
class PerLayer:
    """Loader for <dir>/<model>/[sp<NN>/]{layer_diagnostics,transform_results}.csv."""
    def __init__(self, results_dir, models, sub=""):
        self.dir, self.models = results_dir, models
        self.diag, self.trans = {}, {}
        for m in models:
            dp = os.path.join(results_dir, m, sub, "layer_diagnostics.csv")
            tp = os.path.join(results_dir, m, sub, "transform_results.csv")
            if os.path.exists(dp):
                self.diag[m] = pd.read_csv(dp)
                print(f"  Loaded {m}{('/' + sub) if sub else ''}", file=sys.stderr)
            else:
                print(f"  Missing {m}{('/' + sub) if sub else ''}", file=sys.stderr)
            if os.path.exists(tp):
                self.trans[m] = pd.read_csv(tp)

    def diag_mean(self, model, col, fmt="{:.3f}"):
        if model not in self.diag: return PLACEHOLDER
        df = self.diag[model]
        if col not in df.columns: return PLACEHOLDER
        v = df[col].mean()
        return PLACEHOLDER if pd.isna(v) else fmt.format(v)

    def diag_meanstd(self, model, col, fmt="{:.3f}"):
        if model not in self.diag: return PLACEHOLDER
        df = self.diag[model]
        if col not in df.columns: return PLACEHOLDER
        v, s = df[col].mean(), df[col].std()
        if pd.isna(v): return PLACEHOLDER
        return f"${fmt.format(v)} \\pm {fmt.format(s)}$"

    def delta_mean(self, model, transform, pruning="wanda"):
        if model not in self.trans: return PLACEHOLDER
        t = self.trans[model]
        w = t.loc[(t["pruning"]==pruning) & (t["transform"]==transform), "zero_rows"]
        d = t.loc[(t["pruning"]=="dense")  & (t["transform"]==transform), "zero_rows"]
        if len(w)==0 or len(d)==0: return PLACEHOLDER
        v = (w.mean() - d.mean()) * 100
        return "0.0\\%" if abs(v) < 0.05 else f"{v:+.1f}\\%"

    def dense_zero_rows(self, model, transform):
        if model not in self.trans: return PLACEHOLDER
        t = self.trans[model]
        d = t.loc[(t["pruning"]=="dense") & (t["transform"]==transform), "zero_rows"]
        if len(d)==0: return PLACEHOLDER
        return f"{d.mean()*100:.1f}\\%"

    def layer_delta(self, model, layer_name, transform, pruning="wanda"):
        if model not in self.trans: return PLACEHOLDER
        t = self.trans[model]
        w = t.loc[(t["layer"]==layer_name) & (t["transform"]==transform) &
                  (t["pruning"]==pruning), "zero_rows"]
        d = t.loc[(t["layer"]==layer_name) & (t["transform"]==transform) &
                  (t["pruning"]=="dense"),  "zero_rows"]
        if len(w)==0 or len(d)==0: return PLACEHOLDER
        v = (w.iloc[0] - d.iloc[0]) * 100
        return f"{v:+.1f}\\%"


def _model_header(models):
    return " & ".join(r"\textbf{"+MNAME.get(m,m)+"}" for m in models)


def t_transforms(P):
    # CHANGED: the submitted version listed two transforms the code never
    # implemented (README, "Differences"). Rows now list exactly the 13 in
    # pruning_transforms.transforms.TRANSFORMS (3 + 5 + 2 + 3).
    return [
        r"% Auto-generated by analysis/generate_tables.py",
        r"% \usepackage{booktabs,multirow} required",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Transformations evaluated. $^\dagger$Produce zero rows on"
        r" \emph{any} matrix due to shape or decomposition structure,"
        r" independent of Wanda's pruning.}",
        r"\label{tab:transforms}",
        r"\begin{tabular}{p{3.2cm}p{5.8cm}p{2cm}p{1.8cm}}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Transforms} & \textbf{Acts on} & \textbf{Orthogonal?} \\",
        r"\midrule",
        r"Orthogonal rotations & PCA, Hadamard, DCT & Input & Yes \\[2pt]",
        r"Matrix decompositions & SVD (left, right, two-sided$^\dagger$), QR$^\dagger$, LU & Both & Partially \\[2pt]",
        r"Statistical & Diagonal scaling, SVD weighted & Input & No \\[2pt]",
        r"Permutation & Row sort, Column sort, Cuthill-McKee & Both & Yes \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]


def t_delta(P):
    models = P.models
    cs = "l" + "r"*len(models)
    L = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Wanda's marginal contribution: $\Delta = \mathrm{zero\_rows}("
        r"T(W_\text{wanda})) - \mathrm{zero\_rows}(T(W_\text{dense}))$. "
        r"Std of $\Delta$ across layers $< 0.001$ for all transforms (not shown); non-zero entries are driven by the single \texttt{out\_proj}/\texttt{o\_proj}"
        r" anomaly layer (Table~\ref{tab:outproj}). \textbf{[X]}~=~pending.}",
        r"\label{tab:delta}",
        r"\begin{tabular}{"+cs+"}",
        r"\toprule",
        r"\textbf{Transform} & "+_model_header(models)+r" \\",
        r"\midrule",
    ]
    for tk in TORDER:
        if tk in TBREAKS:
            L.append(r"\midrule")
        vals = " & ".join(P.delta_mean(m, tk) for m in models)
        L.append(TNAME.get(tk,tk) + " & " + vals + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def t_qrsvd(P):
    models = P.models
    cs = "l" + "r"*len(models)
    L = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Zero-row fraction of QR and SVD two-sided applied to"
        r" \emph{dense} unpruned matrices. Values are identical on Wanda-pruned"
        r" matrices ($\Delta=0.0\%$), confirming these structures are intrinsic"
        r" to matrix geometry. QR: $\max(0,C_\text{out}{-}C_\text{in})/C_\text{out}$"
        r" (shape artifact); SVD two-sided: 100\% by construction.}",
        r"\label{tab:qrsvd}",
        r"\begin{tabular}{"+cs+"}",
        r"\toprule",
        r"\textbf{Transform (dense $W$)} & "+_model_header(models)+r" \\",
        r"\midrule",
    ]
    for tk, label in [("QR_left",     r"QR left ($\to R$)"),
                      ("SVD_twosided", r"SVD two-sided ($\Sigma$)")]:
        vals = " & ".join(P.dense_zero_rows(m, tk) for m in models)
        L.append(label + " & " + vals + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def t_diagnostics(P):
    # CHANGED (layout only): the old header declared 9 columns and two
    # malformed header rows, while each body row has 6 cells (R10 and rho(M)
    # are single "mean +- std" cells). Header now matches the body.
    L = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Diagnostic metrics. \emph{rank-drop} $= \mathrm{rank}(W)"
        r"-\mathrm{rank}(W_\text{wanda})$ at $10^{-3}$ and $10^{-6}$."
        r" \emph{hi90} = rank-drop at 90\% magnitude sparsity (positive control)."
        r" $R_{10}$ = top-10 SV energy ratio. $\rho(M)$ = inter-row mask correlation."
        r" Sign reversal across thresholds indicates spectral redistribution. $R_{10}$ and $\rho(M)$ report mean\,$\pm$\,std across sampled layers.}",
        r"\label{tab:diagnostics}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \multicolumn{2}{c}{\textbf{rank-drop}}"
        r" & \textbf{hi90} & $R_{10}$ & $\rho(M)$ \\",
        r"\cmidrule(lr){2-3}",
        r" & $10^{-3}$ & $10^{-6}$ & $10^{-3}$ & mean\,$\pm$\,std & mean\,$\pm$\,std \\",
        r"\midrule",
    ]
    for m in P.models:
        mn      = MNAME.get(m,m)
        rd3     = P.diag_mean(m,"drop_wanda_rank_1em3","{:+.2f}")
        rd6     = P.diag_mean(m,"drop_wanda_rank_1em6","{:+.2f}")
        h90     = P.diag_mean(m,"drop_hi90_rank_1em3", "{:+.2f}")
        r10_ms  = P.diag_meanstd(m,"wanda_top10_ratio","{:.3f}")
        mcw_ms  = P.diag_meanstd(m,"mask_corr_wanda",  "{:.3f}")
        L.append(f"{mn} & {rd3} & {rd6} & {h90} & {r10_ms} & {mcw_ms} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def _short_layer(layer):
    parts = layer.split(".")
    try:
        idx = parts.index("layers")
        return "layers." + parts[idx+1] + r".out\_proj"
    except ValueError:
        return parts[-1]


def t_outproj(P):
    L = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{\texttt{out\_proj}/\texttt{o\_proj} layers with genuine"
        r" rank-drop $>10$ consistent across all thresholds."
        r" SVD-left achieves the largest $\Delta$ of any transform."
        r" Even the strongest case (LLaMA-7B, $\Delta=+10.5\%$) falls far"
        r" below the ${\sim}20$--25\% needed for practical structured pruning.}",
        r"\label{tab:outproj}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Layer} & \multicolumn{2}{c}{\textbf{rank-drop}}"
        r" & $\rho(M)$ & $R_{10}$ & \textbf{eff-rank} & $\Delta$ \\",
        r"\cmidrule(lr){3-4}",
        r" & & $10^{-3}$ & $10^{-6}$ & & & \textbf{change} & \textbf{(SVD-left)} \\",
        r"\midrule",
    ]
    found = False
    for m in P.models:
        if m not in P.diag: continue
        df = P.diag[m]
        if "layer_type" not in df.columns: continue
        out = df.loc[df["layer_type"].isin(["out_proj","o_proj"])].copy()
        dc  = "drop_wanda_rank_1em3"
        if dc not in out.columns: continue
        for _, row in out.loc[out[dc]>10].sort_values(dc,ascending=False).iterrows():
            mn    = MNAME.get(m,m)
            short = _short_layer(row["layer"])
            rd3 = row.get("drop_wanda_rank_1em3", np.nan)
            rd6 = row.get("drop_wanda_rank_1em6", np.nan)
            mc  = row.get("mask_corr_wanda",      np.nan)
            r10 = row.get("wanda_top10_ratio",    np.nan)
            erd = row.get("eff_rank_wanda",np.nan) - row.get("eff_rank_dense",np.nan)
            rd3_s = f"{rd3:+.0f}" if not np.isnan(rd3) else PLACEHOLDER
            rd6_s = f"{rd6:+.0f}" if not np.isnan(rd6) else PLACEHOLDER
            mc_s  = f"{mc:.3f}"   if not np.isnan(mc)  else PLACEHOLDER
            r10_s = f"{r10:.3f}"  if not np.isnan(r10) else PLACEHOLDER
            erd_s = f"{erd:+.0f}" if not np.isnan(erd) else PLACEHOLDER
            dlt   = P.layer_delta(m, row["layer"], "SVD_left")
            L.append(f"{mn} & {short} & {rd3_s} & {rd6_s} & {mc_s} & {r10_s} & {erd_s} & {dlt} \\\\")
            found = True
    if not found:
        L.append(r"\multicolumn{8}{c}{No anomaly layers in completed results.} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def t_rotation(P):
    # NOT FROM A CSV. These values were typed into generate_tables2.py from
    # the output of test_hypothesis.py / test_hypothesis_v2.py. Those scripts
    # are not in this repository (they were not found on the authors'
    # machine), so this table cannot be regenerated. It is reproduced
    # verbatim and flagged.
    rotation_data = [
        ("Identity (no rotation)",   "Baseline: Wanda scores on $W$",                   "0.00\\%", "0.00\\%"),
        ("PCA rotation",             "Right-multiply by activation PCA basis",           "0.06\\%", "0.38\\%"),
        ("Stiefel opt.\\ (output)",  "Left-multiply, entropy minimised on Stiefel mfd.", "0.04\\%", "0.13\\%"),
    ]
    L = [
        "",
        r"% WARNING: hard-coded values (source scripts test_hypothesis*.py missing) -- not reproducible from CSVs",
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Pre-pruning rotation experiment on OPT-125M (10 largest layers)."
        r" Row-importance entropy reduction under orthogonal $Q$ applied"
        r" \emph{before} Wanda scoring."
        r" PCA rotation (activation-variance basis) and Stiefel manifold"
        r" optimisation of $Q$ both fail to concentrate row importance,"
        r" with entropy reductions below 0.1\% on average."
        r" This suggests that activation-variance-based rotation saturates"
        r" quickly and joint optimisation of $Q$ and the pruning mask is needed.}",
        r"\label{tab:rotation}",
        r"\begin{tabular}{lp{5cm}rr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Description} & \textbf{Mean $\Delta H$} & \textbf{Max $\Delta H$} \\",
        r"\midrule",
    ]
    for method, desc, mean, mx in rotation_data:
        L.append(f"{method} & {desc} & {mean} & {mx} \\\\")
    L += [
        r"\bottomrule",
        r"\multicolumn{4}{l}{\small $\Delta H = $ normalised entropy reduction of"
        r" Wanda row-importance scores. Higher = more concentrated.} \\",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return L


def t_appendix(P):
    # REPORTED, NOT FIXED: for the Exact-rows rows
    # this compares Wanda zero_rows at threshold t (sensitivity.csv) against
    # DENSE zero_rows at the fixed 1e-6 threshold (transform_results.csv); the
    # dense side is never re-thresholded, because the pipeline does not store
    # dense zero_rows at 1e-3/1e-4. The 1e-5 row is always [X]: the pipeline
    # sweeps only SENSITIVITY_THRESHOLDS = [1e-3, 1e-4, 1e-6]. The caption's
    # "fully zero" 4x4 blocks are blocks that are >= 90% zero (BLOCK_THRESH).
    # Reproduced exactly as published pending the authors' decision.
    models = P.models
    L = [
        "",
        r"% ── APPENDIX ─────────────────────────────────────────────────────────────",
        r"\section*{Appendix}",
        r"\setcounter{table}{0}",
        r"\renewcommand{\thetable}{A\arabic{table}}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Wanda's marginal contribution $\Delta$ across all four sparsity"
        r" metrics and all thresholds, averaged across transforms and layers."
        r" All entries are $\leq 0.1\%$, confirming that $\Delta \approx 0$ is"
        r" not specific to the exact-row metric or any particular threshold."
        r" \emph{Exact}: fraction of fully-zero rows."
        r" \emph{Near-zero}: fraction of rows with norm $< 1\%$ of mean row norm."
        r" \emph{Block}: fraction of $4{\times}4$ blocks that are fully zero."
        r" \emph{Column}: fraction of fully-zero columns."
        r" Values are mean $\Delta$ (wanda $-$ dense) across all 14 transforms"
        r" and all sampled layers per model.}",
        r"\label{tab:app_metrics}",
        r"\begin{tabular}{ll" + "r"*len(models) + "}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Threshold} & " + _model_header(models) + r" \\",
        r"\midrule",
    ]
    METRICS = [("Exact rows", "zero_rows"), ("Near-zero", "near_zero_rows"),
               ("Block 4x4", "block4x4"), ("Columns", "zero_cols")]
    THRESH_VALS = [1e-3, 1e-4, 1e-5, 1e-6]
    THRESH_STRS = [r"$10^{-3}$", r"$10^{-4}$", r"$10^{-5}$", r"$10^{-6}$"]
    for m_idx, (metric_label, metric_col) in enumerate(METRICS):
        if m_idx > 0:
            L.append(r"\midrule")
        if metric_col == "zero_rows":
            for t_val, t_str in zip(THRESH_VALS, THRESH_STRS):
                row_vals = []
                for m in models:
                    sens_path = os.path.join(P.dir, m, "sensitivity.csv")
                    if not os.path.exists(sens_path):
                        row_vals.append(PLACEHOLDER); continue
                    sens = pd.read_csv(sens_path)
                    w = sens.loc[(sens["threshold"].round(10) == round(t_val, 10)), "zero_rows"]
                    if m not in P.trans:
                        row_vals.append(PLACEHOLDER); continue
                    t_df = P.trans[m]
                    d = t_df.loc[t_df["pruning"]=="dense", "zero_rows"]
                    if len(w)==0 or len(d)==0:
                        row_vals.append(PLACEHOLDER); continue
                    row_vals.append(f"{(w.mean() - d.mean()) * 100:+.2f}\\%")
                prefix = metric_label if t_val == THRESH_VALS[0] else ""
                L.append(f"{prefix} & {t_str} & " + " & ".join(row_vals) + r" \\")
        else:
            row_vals = []
            for m in models:
                if m not in P.trans or metric_col not in P.trans[m].columns:
                    row_vals.append(PLACEHOLDER); continue
                t_df = P.trans[m]
                w = t_df.loc[t_df["pruning"]=="wanda", metric_col]
                d = t_df.loc[t_df["pruning"]=="dense",  metric_col]
                if len(w)==0 or len(d)==0:
                    row_vals.append(PLACEHOLDER); continue
                row_vals.append(f"{(w.mean() - d.mean()) * 100:+.2f}\\%")
            L.append(f"{metric_label} & — & " + " & ".join(row_vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


# ══════════════════════════════════════════════════════════════════════════════
# PER-LAYER SCHEMA — rebuttal sweep (<dir>/<model>/sp<NN>/). Ported from
# generate_rebuttal_tables.py; output byte-identical.
# ══════════════════════════════════════════════════════════════════════════════
def rebuttal_tables(results_dir, models, sparsities, sg):
    loaders = {sp: PerLayer(results_dir, models, sub=f"sp{sp}") for sp in sparsities}
    if sg not in loaders:
        loaders[sg] = PerLayer(results_dir, models, sub=f"sp{sg}")

    def has_sgpt(m, sp):
        t = loaders[sp].trans.get(m)
        return t is not None and "sparsegpt" in t["pruning"].unique()

    L = []
    for m in models:
        cs  = "l" + "r" * len(sparsities)
        sph = " & ".join(r"\textbf{" + f"{sp}\\%" + "}" for sp in sparsities)
        L += [
            r"% Auto-generated by generate_rebuttal_tables.py",
            r"\begin{table}[ht]",
            r"\centering",
            r"\small",
            r"\caption{Sparsity ablation for " + MNAME.get(m, m) + r": "
            r"Wanda's marginal contribution $\Delta$ across moderate sparsity "
            r"levels 50--80\%. Construction identical to the main-paper $\Delta$ "
            r"metric (Table~2), recomputed at each sparsity level. "
            r"\textbf{[X]}~=~not yet run.}",
            r"\label{tab:sparsity_ablation_" + m.replace(".", "p").replace("-", "_") + "}",
            r"\begin{tabular}{" + cs + "}",
            r"\toprule",
            r"\textbf{Transform} & " + sph + r" \\",
            r"\midrule",
        ]
        for tk in TORDER:
            if tk in TBREAKS:
                L.append(r"\midrule")
            vals = " & ".join(loaders[sp].delta_mean(m, tk, "wanda") for sp in sparsities)
            L.append(TNAME.get(tk, tk) + " & " + vals + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    with_sgpt = [m for m in models if has_sgpt(m, sg)]
    if with_sgpt:
        cs2 = "l" + "rr" * len(with_sgpt)
        mh2 = " & ".join(r"\multicolumn{2}{c}{\textbf{" + MNAME.get(m, m) + "}}" for m in with_sgpt)
        sub2 = " & ".join(r"Wanda & SparseGPT" for _ in with_sgpt)
        L += [
            r"\begin{table}[ht]",
            r"\centering",
            r"\small",
            r"\caption{Wanda vs.\ SparseGPT marginal contribution $\Delta$ at "
            f"{sg}\\% sparsity. SparseGPT uses the official Hessian-based "
            r"sequential weight update (\citet{frantar2023sparsegpt}); "
            r"$\Delta$ is constructed identically for both "
            r"($\mathrm{zero\_rows}(T(W_\text{pruned})) - \mathrm{zero\_rows}(T(W_\text{dense}))$). "
            r"Agreement between the two columns indicates the null result is "
            r"a property of per-output pruning broadly, not an artifact of "
            r"Wanda's specific scoring rule. \textbf{[X]}~=~not yet run.}",
            r"\label{tab:sparsegpt_comparison}",
            r"\begin{tabular}{" + cs2 + "}",
            r"\toprule",
            r" & " + mh2 + r" \\",
            r"\textbf{Transform} & " + sub2 + r" \\",
            r"\midrule",
        ]
        for tk in TORDER:
            if tk in TBREAKS:
                L.append(r"\midrule")
            cells = []
            for m in with_sgpt:
                cells.append(loaders[sg].delta_mean(m, tk, "wanda"))
                cells.append(loaders[sg].delta_mean(m, tk, "sparsegpt"))
            L.append(TNAME.get(tk, tk) + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    else:
        L += [r"% No models have completed SparseGPT runs at "
              f"{sg}\\% sparsity -- Table R2 omitted.", ""]

    L += [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{\texttt{out\_proj}/\texttt{o\_proj} anomaly layers (flagged via "
        r"Wanda rank-drop $>10$ at $" + f"{sg}" + r"$\% sparsity), compared against "
        r"SparseGPT on the \emph{same layers}. If the anomaly is intrinsic to the "
        r"layer's weight geometry, SparseGPT should show comparable rank-drop and "
        r"$\rho(M)$; if it is specific to Wanda's per-output comparison group, "
        r"SparseGPT's jointly-optimized update should not reproduce it. "
        r"\textbf{[X]}~=~not yet run.}",
        r"\label{tab:outproj_sgpt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Layer} & \multicolumn{2}{c}{\textbf{rank-drop @1e-3}}"
        r" & \multicolumn{2}{c}{$\rho(M)$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & Wanda & SparseGPT & Wanda & SparseGPT \\",
        r"\midrule",
    ]
    found = False
    for m in models:
        df = loaders[sg].diag.get(m)
        if df is None or "layer_type" not in df.columns or "drop_wanda_rank_1em3" not in df.columns:
            continue
        anomaly = df.loc[df["layer_type"].isin(["out_proj", "o_proj"])
                         & (df["drop_wanda_rank_1em3"] > 10)
                         ].sort_values("drop_wanda_rank_1em3", ascending=False)
        for _, row in anomaly.iterrows():
            rd_w = row.get("drop_wanda_rank_1em3", np.nan)
            rd_s = row.get("drop_sgpt_rank_1em3", np.nan) if "drop_sgpt_rank_1em3" in df.columns else np.nan
            mc_w = row.get("mask_corr_wanda", np.nan)
            mc_s = row.get("mask_corr_sgpt", np.nan) if "mask_corr_sgpt" in df.columns else np.nan
            f0 = lambda v: f"{v:+.0f}" if not pd.isna(v) else PLACEHOLDER
            f3 = lambda v: f"{v:.3f}" if not pd.isna(v) else PLACEHOLDER
            L.append(f"{MNAME.get(m, m)} & {_short_layer(row['layer'])} & {f0(rd_w)} & {f0(rd_s)} & {f3(mc_w)} & {f3(mc_s)} \\\\")
            found = True
    if not found:
        L.append(r"\multicolumn{6}{c}{No anomaly layers in completed results.} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


# ══════════════════════════════════════════════════════════════════════════════
# WHOLE-MODEL SCHEMA (<dir>/<model>/<method>/sp<NN>/seed<S>/)
# ══════════════════════════════════════════════════════════════════════════════
def load_wholemodel(results_dir):
    """All delta_per_transform.csv + run_info.json, long format."""
    rows, infos = [], []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "sp*", "seed*",
                                              "delta_per_transform.csv"))):
        d = os.path.dirname(path)
        seed = int(os.path.basename(d)[4:]); sp = os.path.basename(os.path.dirname(d))
        method = os.path.basename(os.path.dirname(os.path.dirname(d)))
        model = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(d))))
        df = pd.read_csv(path)
        df["model"], df["method"], df["sp"], df["seed"] = model, method, sp, seed
        rows.append(df)
        ip = os.path.join(d, "run_info.json")
        if os.path.exists(ip):
            info = json.load(open(ip)); info["sp"] = sp; info["dir"] = d
            infos.append(info)
    if not rows:
        sys.exit(f"no whole-model delta_per_transform.csv under {results_dir}")
    return pd.concat(rows, ignore_index=True), infos


def t_methods(results_dir, model, sp, metric):
    """Delta(metric) per transform for every whole-model method of one model,
    mean over calibration seeds (+- std when >1 seed)."""
    df, _ = load_wholemodel(results_dir)
    df = df[(df["model"] == model) & (df["sp"] == sp)]
    methods = [m for m in ["wanda", "sparsegpt", "magnitude_perrow", "magnitude"]
               if m in df["method"].unique()]
    seeds = {m: sorted(df.loc[df["method"] == m, "seed"].unique()) for m in methods}
    label = {"zero_rows": "zero\\_rows", "zero_cols": "zero\\_cols", "block4x4": "block4x4"}[metric]
    L = [
        r"% Auto-generated by analysis/generate_tables.py (whole-model protocol)",
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{Whole-model protocol, {MNAME.get(model, model)} at {sp[2:]}\% sparsity: "
        rf"$\Delta = \mathrm{{{label}}}(T(W_\text{{pruned}})) - \mathrm{{{label}}}(T(W_\text{{dense}}))$ "
        r"per pruning method (official Wanda/SparseGPT code; per-row magnitude). "
        r"Mean over calibration seeds "
        + ", ".join(f"{METHOD_NAME.get(m, m)}: {{{','.join(map(str, s))}}}" for m, s in seeds.items())
        + r"; $\pm$ = std across seeds.}",
        rf"\label{{tab:wholemodel_{metric}_{model.replace('.', 'p').replace('-', '_')}}}",
        r"\begin{tabular}{l" + "r" * len(methods) + "}",
        r"\toprule",
        r"\textbf{Transform} & " + " & ".join(r"\textbf{" + METHOD_NAME.get(m, m) + "}" for m in methods) + r" \\",
        r"\midrule",
    ]
    for tk in TORDER:
        if tk in TBREAKS:
            L.append(r"\midrule")
        cells = []
        for m in methods:
            v = df.loc[(df["method"] == m) & (df["transform"] == tk), metric]
            if len(v) == 0:
                cells.append(PLACEHOLDER)
            elif len(v) > 1:
                cells.append(f"${v.mean()*100:+.2f} \\pm {v.std()*100:.2f}$\\%")
            else:
                cells.append(f"{v.iloc[0]*100:+.2f}\\%")
        L.append(TNAME.get(tk, tk) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def t_seeds(results_dir):
    """E1: Delta(zero_rows) variance over calibration seeds."""
    p = os.path.join(results_dir, "seed_variance_summary.csv")
    if not os.path.exists(p):
        sys.exit(f"missing {p} (written by run_wholemodel_experiments.py)")
    df = pd.read_csv(p)
    df = df[(df["metric"] == "zero_rows") & (df["n_seeds"] > 1)]
    L = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Calibration-seed variance of $\Delta(\mathrm{zero\_rows})$ (whole-model protocol). "
        r"One-sample $t$-test against $\Delta=0$; -- = zero variance across seeds.}",
        r"\label{tab:seed_variance}",
        r"\begin{tabular}{lllrrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{Method} & \textbf{Transform} & $n$ & \textbf{mean} & \textbf{std} & $p$ \\",
        r"\midrule",
    ]
    order = {t: i for i, t in enumerate(TORDER)}
    df = df.assign(_o=df["transform"].map(order)).sort_values(["model", "method", "_o"])
    for _, r in df.iterrows():
        p_s = "--" if pd.isna(r["p_value"]) else f"{r['p_value']:.3f}"
        L.append(f"{MNAME.get(r['model'], r['model'])} & {METHOD_NAME.get(r['method'], r['method'])} & "
                 f"{TNAME.get(r['transform'], r['transform'])} & {int(r['n_seeds'])} & "
                 f"{r['delta_mean']*100:+.3f}\\% & {r['delta_std']*100:.3f}\\% & {p_s} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def t_ppl(results_dir):
    """Perplexity sanity check of the official pruning (and E3 row deletion)."""
    _, infos = load_wholemodel(results_dir)
    infos = [i for i in infos if "ppl_wikitext2_pruned" in i]
    L = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{WikiText-2 perplexity after whole-model pruning with the official code, "
        r"and after SVD-left-guided rank truncation (row deletion) of one attention output projection. "
        r"Reference: Wanda, LLaMA-7B, 50\%: 7.26 \citep{sun2024wanda}.}",
        r"\label{tab:ppl_rowdel}",
        r"\begin{tabular}{llrrlrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{Method} & \textbf{seed} & \textbf{PPL} & \textbf{truncated layer} & "
        r"\textbf{rank kept} & \textbf{rows deleted} & \textbf{PPL after} \\",
        r"\midrule",
    ]
    for i in infos:
        t = i.get("truncate_oproj")
        tr = (f"{t['layer'].replace('_', chr(92) + '_')} & {t['rank_kept']}/{min(t['C_out'], t['C_in'])} & "
              f"{t['rows_deleted']} & {i['ppl_wikitext2_truncated']:.4f}") if t else "-- & -- & -- & --"
        L.append(f"{MNAME.get(i['model'], i['model'])} & {METHOD_NAME.get(i['method'], i['method'])} & "
                 f"{i['seed']} & {i['ppl_wikitext2_pruned']:.4f} & {tr} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def t_outproj_perrow(results_dir, model="opt-125m", sp="sp50", seed=0):
    """Per-row magnitude control on out_proj (whole-model): mean rank drop,
    rho(M) and Delta(SVD-left zero_rows). Formerly computed ad hoc in
    rock.py, whose Delta line used `t.transform == "SVD_left"` (the DataFrame
    METHOD, not the column) and printed nan; the correct value is below."""
    d = os.path.join(results_dir, model, "magnitude_perrow", sp, f"seed{seed}")
    t = pd.read_csv(os.path.join(d, "transform_results.csv"))
    g = pd.read_csv(os.path.join(d, "layer_diagnostics.csv"))
    o = t[t["layer"].str.contains("out_proj") & (t["transform"] == "SVD_left")]
    delta = (o.loc[o["pruning"] == "magnitude_perrow", "zero_rows"].mean()
             - o.loc[o["pruning"] == "dense", "zero_rows"].mean())
    og = g[g["layer"].str.contains("out_proj")]
    return [
        f"% {MNAME.get(model, model)} per-row magnitude, {sp}, seed {seed}, out_proj layers (whole-model)",
        f"% mean rank-drop @1e-3 = {og['drop_rank_1em3'].mean():+.3f}",
        f"% mean rho(M)          = {og['mask_corr'].mean():.4f}",
        f"% Delta(SVD-left) zero_rows = {delta*100:+.4f}%  ({delta:.6g})",
        "",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# OTHER EXPERIMENTS (own CSV layouts)
# ══════════════════════════════════════════════════════════════════════════════
def t_crosslayer(results_dir):
    """E2: SliceGPT-style cross-layer rotation, per model/method/q_mode."""
    L = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Cross-layer (SliceGPT-style) rotation $Q$ applied post hoc to whole-model-pruned weights. "
        r"$\Delta$ = mean over sampled layers of $m(QW_\text{pruned}) - m(QW_\text{dense})$; max in parentheses.}",
        r"\label{tab:crosslayer_q}",
        r"\begin{tabular}{lllrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{Method} & $Q$ & $\Delta$ zero rows & $\Delta$ near-zero rows & $\Delta$ near-zero cols \\",
        r"\midrule",
    ]
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "*", "sp*", "run_info.json"))):
        i = json.load(open(p))
        for q, s in i["delta_summary"].items():
            cell = lambda k: f"{s[f'delta_{k}_mean']*100:+.2f}\\% ({s[f'delta_{k}_max']*100:+.1f}\\%)"
            L.append(f"{MNAME.get(i['model'], i['model'])} & {METHOD_NAME.get(i['method'], i['method'])} & "
                     f"{q.replace('_', chr(92) + '_')} & {cell('zero_rows')} & {cell('near_zero_rows')} & {cell('near_zero_cols')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def t_oproj_mechanism(results_dir):
    """E4: out_proj anomaly mechanism, target vs controls."""
    L = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Mechanism of the \texttt{out\_proj}/\texttt{o\_proj} anomaly (whole-model pruning, all blocks). "
        r"Across-head share of activation-norm variance, mask correlation $\rho(M)$, Spearman correlation of head norm "
        r"vs head zero rate, dead-column fraction, rank drop, $R_{10}$ and $\Delta$(SVD-left), averaged over depth.}",
        r"\label{tab:oproj_mechanism}",
        r"\begin{tabular}{llllrrrrrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{Method} & \textbf{Layer} & \textbf{Role} & across & $\rho(M)$ & Spearman & dead cols & rank-drop & $R_{10}$ & $\Delta$ \\",
        r"\midrule",
    ]
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "*", "sp*", "summary_by_layer_type.csv"))):
        method = os.path.basename(os.path.dirname(os.path.dirname(p)))
        model  = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            L.append(f"{MNAME.get(model, model)} & {METHOD_NAME.get(method, method)} & "
                     f"{r['layer_type'].replace('_', chr(92) + '_')} & {r['role'].replace('_', ' ')} & "
                     f"{r['xn_across_head_share']:.3f} & {r['rho_M']:.3f} & {r['spearman_headnorm_zerorate']:+.3f} & "
                     f"{r['frac_dead_cols']:.3f} & {r['rank_drop_1em3']:+.1f} & {r['R10']:.3f} & "
                     f"{r['delta_svdleft_zero_rows']*100:+.2f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def t_synthetic(results_dir):
    """E5: synthetic three-mechanism factorial."""
    p = os.path.join(results_dir, "synthetic_mechanism_ablation_summary.csv")
    if not os.path.exists(p):
        sys.exit(f"missing {p} (run experiments/synthetic_mechanism_ablation.py)")
    df = pd.read_csv(p)
    L = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Synthetic factorial over the three mechanisms ($n=512$, 50\% sparsity, 5 seeds). "
        r"Mask: independent per-row (Wanda-like), $G$ row groups sharing a mask, or a fraction $c$ of columns zeroed in every row. "
        r"$\Delta$ = SVD-left zero-row gain (mean $\pm$ std over seeds); best = max over transforms.}",
        r"\label{tab:synthetic}",
        r"\begin{tabular}{lllrrrrr}", r"\toprule",
        r"\textbf{Mask} & \textbf{Spectrum} & \textbf{Rank} & $\rho(M)$ & rank-drop & $R_{10}$ & $\Delta$ (SVD-left) & best $\Delta$ \\",
        r"\midrule",
    ]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r["mask"] != prev:
            L.append(r"\midrule")
        prev = r["mask"]
        L.append(f"{r['mask'].replace('_', chr(92) + '_')} & {r['spectrum']} & {r['rank']} & {r['rho_M']:.3f} & "
                 f"{r['rank_drop']:.1f} & {r['R10']:.3f} & "
                 f"${r['delta_svdleft_rows']:+.3f} \\pm {r['delta_svdleft_rows_std']:.3f}$ & {r['best_delta_rows']:+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA-READY TABLES (numbered as in the camera-ready PDF)
#
# These combine protocols (e.g. Table 2 shows per-layer values with whole-model
# values in brackets), so they read from one results root laid out as the
# experiment scripts write it:
#   <root>/perlayer/  <root>/wholemodel/  <root>/crosslayer_q/
#   <root>/oproj_mechanism/  <root>/synthetic/
# Nothing is recomputed; only selection, averaging over seeds/layers and
# formatting happen here.
# ══════════════════════════════════════════════════════════════════════════════
CR_MODELS = ["opt-125m", "opt-1.3b", "opt-6.7b", "llama-7b", "llama-13b"]


def _half_up(v, nd):
    """Round half away from zero (-9.25 -> -9.3), as the paper's tables do;
    Python's format() would print -9.2."""
    from decimal import Decimal, ROUND_HALF_UP
    return Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def _p2(v):
    """Whole-model percentage cell: exact zero prints as 0.00%, otherwise signed."""
    return "0.00\\%" if v == 0 else f"{v*100:+.2f}\\%"


def _cr_short_layer(layer, prefix="layers."):
    parts = layer.split(".")
    i = parts.index("layers")
    return f"{prefix}{parts[i+1]}.{parts[-1]}".replace("_", r"\_")


def _wm_delta(root, model, method, sp="sp50"):
    """Whole-model Delta(zero_rows) per transform, mean over available seeds."""
    paths = glob.glob(os.path.join(root, "wholemodel", model, method, sp, "seed*",
                                   "delta_per_transform.csv"))
    if not paths:
        return None, 0
    d = pd.concat([pd.read_csv(p, index_col=0) for p in paths])
    return d.groupby(level=0)["zero_rows"].mean(), len(paths)


def _mech(root, model, method="wanda", sp="sp50"):
    p = os.path.join(root, "oproj_mechanism", model, method, sp, "layer_mechanism.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _table(caption, label, colspec, header, body, extra_header=()):
    return ([r"\begin{table}[ht]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + colspec + "}", r"\toprule", header + r" \\"]
            + list(extra_header) + [r"\midrule"] + body
            + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def cr_table1(root):
    return _table(
        r"The 13 linear transformations evaluated, organised by family, together with the"
        r" learned cross-layer rotations evaluated separately. $^\dagger$These produce zero rows"
        r" on any matrix as a consequence of shape or decomposition structure, independent of"
        r" pruning. $^\ddagger$Derived from the residual-stream activation covariance of the"
        r" pruned model and applied across layers by computational invariance; the Haar-random"
        r" variant serves as a control.",
        "tab:transforms", "p{2.8cm}p{5.6cm}p{2.2cm}p{1.8cm}",
        r"\textbf{Family} & \textbf{Transforms} & \textbf{Acts on} & \textbf{Orthogonal?}",
        [r"Orthogonal rot. & PCA, Hadamard, DCT & Input (right) & Yes \\",
         r"Matrix decomp. & SVD (left, right, two-sided$^\dagger$), QR$^\dagger$, LU & Both & Partially \\",
         r"Statistical & Diagonal scaling, SVD weighted & Input & No \\",
         r"Permutation & Row sort, Column sort, Cuthill-McKee & Both & Yes (discrete) \\",
         r"\midrule",
         r"Learned, cross-layer$^\ddagger$ & Global $Q$, per-block $Q_i$, Haar-random $Q$"
         r" & Stream (both sides) & Yes \\"])


def cr_table2(root):
    """Per-layer Delta with whole-model values in brackets."""
    pl = os.path.join(root, "perlayer")
    P = PerLayer(pl, CR_MODELS)
    # Main-sweep LLaMA CSVs have no Hadamard rows; the 50% rebuttal rerun does.
    P50 = PerLayer(pl, CR_MODELS, sub="sp50")
    wm = {m: _wm_delta(root, m, "wanda") for m in CR_MODELS}
    body, notes = [], []
    for tk in TORDER:
        if tk in TBREAKS:
            body.append(r"\midrule")
        cells = []
        for m in CR_MODELS:
            c = P.delta_mean(m, tk)
            if c == PLACEHOLDER and P50.delta_mean(m, tk) != PLACEHOLDER:
                c = P50.delta_mean(m, tk)
                notes.append(f"% {m} {tk}: from perlayer/{m}/sp50 (main sweep has no rows)")
            d, _ = wm[m]
            if d is not None and tk in d.index:
                c += f" ({_p2(d[tk])})"
            cells.append(c)
        body.append(TNAME.get(tk, tk) + " & " + " & ".join(cells) + r" \\")
    seeds = {m: n for m, (_, n) in wm.items() if n}
    cap = (r"Marginal contribution of pruning: $\Delta = \mathrm{zero\_rows}(T(W_\text{wanda}))"
           r" - \mathrm{zero\_rows}(T(W_\text{dense}))$. Values are from the per-layer protocol;"
           r" where whole-model sequential pruning was also run, those values appear in brackets"
           r" (mean over " + ", ".join(f"{MNAME[m]}: {n} draw{'s' if n > 1 else ''}"
                                        for m, n in seeds.items()) + r")."
           r" Non-zero entries are driven by the single \texttt{out\_proj}/\texttt{o\_proj}"
           r" anomaly layer (Table~\ref{tab:outproj}).")
    cs = "l" + "r" * len(CR_MODELS)
    return notes + _table(cap, "tab:delta", cs, r"\textbf{Transform} & " + _model_header(CR_MODELS), body)


def _methods_frame(root, model="opt-125m"):
    cols = {}
    for m in ["wanda", "sparsegpt", "magnitude_perrow"]:
        d, n = _wm_delta(root, model, m)
        if d is not None:
            cols[m] = d
    return pd.DataFrame(cols)


def _methods_header(df):
    return r"\textbf{Transform} & " + " & ".join(
        r"\textbf{" + {"magnitude_perrow": "Magnitude (per-row)"}.get(m, METHOD_NAME[m]) + "}"
        for m in df.columns)


def cr_table3(root):
    df = _methods_frame(root)
    rows = [("Identity", TNAME["Identity"]), ("SVD_left", TNAME["SVD_left"]),
            ("Hadamard_right", TNAME["Hadamard_right"])]
    body = [lab + " & " + " & ".join(_p2(df.loc[t, m]) for m in df.columns) + r" \\" for t, lab in rows]
    mx = df.drop(index="Identity").max()
    body += [r"\midrule", "Maximum over 13 transforms & " + " & ".join(_p2(mx[m]) for m in df.columns) + r" \\"]
    cap = (r"Generality across per-output pruning methods: $\Delta = \mathrm{zero\_rows}(T(W_\text{pruned}))"
           r" - \mathrm{zero\_rows}(T(W_\text{dense}))$ on OPT-125M at 50\% sparsity under whole-model"
           r" pruning. Three representative transforms are shown: the identity baseline, the transform"
           r" attaining the largest $\Delta$, and one orthogonal rotation standing for the remaining ten."
           r" The final row is the maximum over all 13 transforms. Full per-transform results are in Appendix B.")
    return _table(cap, "tab:methods", "l" + "r" * len(df.columns), _methods_header(df), body)


def cr_a3(root):
    df = _methods_frame(root)
    body = []
    for tk in TORDER:
        body.append(TNAME.get(tk, tk) + " & " + " & ".join(_p2(df.loc[tk, m]) for m in df.columns) + r" \\")
    mx = df.drop(index="Identity").max()
    body += [r"\midrule", "Maximum & " + " & ".join(_p2(mx[m]) for m in df.columns) + r" \\"]
    cap = (r"Complete per-transform $\Delta$ (exact zero rows) on OPT-125M at 50\% sparsity under"
           r" whole-model pruning. Wanda and SparseGPT values are means over three calibration draws;"
           r" per-row magnitude pruning is deterministic and uses a single run.")
    return _table(cap, "tab:app_methods", "l" + "r" * len(df.columns), _methods_header(df), body)


def cr_table5(root):
    Q = [("global", "Global $Q$"), ("per_block", "Per-block $Q_i$"), ("random", "Haar-random $Q$")]
    body = []
    for method in ["wanda", "sparsegpt"]:
        p = glob.glob(os.path.join(root, "crosslayer_q", "opt-125m", method, "sp50", "run_info.json"))
        if not p:
            continue
        s = json.load(open(p[0]))["delta_summary"]
        if body:
            body.append(r"\midrule")
        for i, (q, lab) in enumerate(Q):
            first = (r"\multirow{3}{*}{" + METHOD_NAME[method] + "}") if i == 0 else ""
            body.append(f"{first} & {lab} & " + " & ".join(
                _p2(s[q][f"delta_{k}_mean"]) for k in ("zero_rows", "near_zero_rows", "near_zero_cols")) + r" \\")
    cap = (r"Learned cross-layer rotations on whole-model-pruned OPT-125M at 50\% sparsity. $\Delta$ is"
           r" measured against the same rotation applied to the dense weights. The covariance-derived"
           r" rotations are indistinguishable from a Haar-random control on every metric.")
    return _table(cap, "tab:crosslayer", "llrrr",
                  r"\textbf{Pruning} & \textbf{Rotation} & $\Delta$ zero rows & $\Delta$ near-zero rows & $\Delta$ near-zero cols",
                  body)


def _anomaly_rows(root):
    """Per-layer out_proj/o_proj rows with Wanda rank-drop > 10, joined to the
    whole-model mechanism run (seed 0) where it exists. Also returns the
    layers that pass the criterion only under whole-model pruning."""
    P = PerLayer(os.path.join(root, "perlayer"), CR_MODELS)
    rows, seen = [], set()
    for m in CR_MODELS:
        if m not in P.diag:
            continue
        df = P.diag[m]
        out = df[df["layer_type"].isin(["out_proj", "o_proj"]) & (df["drop_wanda_rank_1em3"] > 10)]
        mech = _mech(root, m)
        for _, r in out.iterrows():
            w = None
            if mech is not None:
                hit = mech[mech["layer"] == r["layer"]]
                w = hit.iloc[0] if len(hit) else None
            rows.append(dict(model=m, layer=r["layer"], wm_only=False, pl=r, wm=w,
                             delta=P.layer_delta(m, r["layer"], "SVD_left")))
            seen.add(r["layer"])
        if mech is not None:
            t = mech[(mech["role"] == "target") & (mech["rank_drop_1em3"] > 10)]
            for _, w in t.iterrows():
                if w["layer"] not in seen:
                    rows.append(dict(model=m, layer=w["layer"], wm_only=True, pl=None, wm=w, delta=None))
    blk = lambda r: int(r["layer"].split(".")[r["layer"].split(".").index("layers") + 1])
    rows.sort(key=lambda r: (CR_MODELS.index(r["model"]), blk(r)))
    return rows


def _pair(pl, wm, fmt):
    a = fmt(pl) if pl is not None else ""
    b = f"({fmt(wm)})" if wm is not None else ""
    return (a + " " + b).strip()


def cr_table6(root):
    body = []
    for r in _anomaly_rows(root):
        if r["wm_only"]:
            continue
        pl, w = r["pl"], r["wm"]
        get = lambda k: None if w is None else w[k]
        body.append(f"{MNAME[r['model']]} & {_cr_short_layer(r['layer'])} & "
                    f"{_pair(pl['drop_wanda_rank_1em3'], get('rank_drop_1em3'), lambda v: f'{v:+.0f}')} & "
                    f"{_pair(pl['drop_wanda_rank_1em6'], get('rank_drop_1em6'), lambda v: f'{v:+.0f}')} & "
                    f"{_pair(pl['mask_corr_wanda'], get('rho_M'), lambda v: f'{v:.3f}')} & "
                    f"{r['delta']}" + (f" ({w['delta_svdleft_zero_rows']*100:+.1f}\\%)" if w is not None else "")
                    + r" \\")
    cap = (r"\texttt{out\_proj}/\texttt{o\_proj} layers with genuine rank-drop $>10$. Values are from the"
           r" per-layer protocol, with whole-model sequential pruning in brackets. Spectral and"
           r" effective-rank diagnostics are in Appendix C.")
    return _table(cap, "tab:outproj", "llrrrr",
                  r"\textbf{Model} & \textbf{Layer} & \multicolumn{2}{c}{\textbf{rank-drop}} & $\rho(M)$ & $\Delta$",
                  body, [r"\cmidrule(lr){3-4}", r" & & $10^{-3}$ & $10^{-6}$ & & (SVD-left) \\"])


def cr_a4(root):
    body = []
    for r in _anomaly_rows(root):
        pl, w = r["pl"], r["wm"]
        get = lambda k: None if w is None else w[k]
        plg = lambda k: None if pl is None else pl[k]
        mn = MNAME[r["model"]] + (r"$^\dagger$" if r["wm_only"] else "")
        body.append(f"{mn} & {_cr_short_layer(r['layer'], 'l.')} & "
                    f"{_pair(plg('drop_wanda_rank_1em3'), get('rank_drop_1em3'), lambda v: f'{v:+.0f}')} & "
                    f"{_pair(plg('drop_wanda_rank_1em6'), get('rank_drop_1em6'), lambda v: f'{v:+.0f}')} & "
                    f"{_pair(plg('mask_corr_wanda'), get('rho_M'), lambda v: f'{v:.3f}')} & "
                    f"{_pair(plg('wanda_top10_ratio'), get('R10'), lambda v: f'{v:.3f}')} \\\\")
    cap = (r"Rank and spectral diagnostics for \texttt{out\_proj}/\texttt{o\_proj} layers with rank-drop"
           r" $>10$. Per-layer protocol, with whole-model sequential pruning in brackets. Rows marked"
           r" $^\dagger$ reach the criterion only under whole-model pruning. Transform results for the"
           r" same layers are in Table~\ref{tab:app_outproj_delta}.")
    return _table(cap, "tab:app_outproj_diag", "llrrrr",
                  r"\textbf{Model} & \textbf{Layer} & \multicolumn{2}{c}{\textbf{rank-drop}} & $\rho(M)$ & $R_{10}$",
                  body, [r"\cmidrule(lr){3-4}", r" & & $10^{-3}$ & $10^{-6}$ & & \\"])


def cr_a5(root):
    body = []
    for r in _anomaly_rows(root):
        pl, w = r["pl"], r["wm"]
        erd = "---" if pl is None else f"{pl['eff_rank_wanda'] - pl['eff_rank_dense']:+.0f}"
        wd = f"({w['delta_svdleft_zero_rows']*100:+.1f}\\%)" if w is not None else ""
        mn = MNAME[r["model"]] + (r"$^\dagger$" if r["wm_only"] else "")
        body.append(f"{mn} & {_cr_short_layer(r['layer'], 'l.')} & {erd} & "
                    f"{((r['delta'] or '') + ' ' + wd).strip()} \\\\")
    cap = (r"Effective-rank change and marginal contribution under SVD-left for the layers of"
           r" Table~\ref{tab:app_outproj_diag}. Effective rank is the entropy of the normalised"
           r" singular-value distribution; it was not recomputed under the whole-model protocol.")
    return _table(cap, "tab:app_outproj_delta", "llrr",
                  r"\textbf{Model} & \textbf{Layer} & \textbf{eff-rank change} & $\Delta$ (SVD-left)", body)


def cr_a1(root):
    # Same computation as t_appendix (see the note there); the camera-ready
    # table omits the 10^-5 row, which the sensitivity sweep never produced.
    P = PerLayer(os.path.join(root, "perlayer"), CR_MODELS)
    L = t_appendix(P)
    L = [l for l in L if not l.lstrip().startswith(r"& $10^{-5}$")]
    i = L.index(r"\begin{table}[ht]")
    return L[i:]


def cr_a6(root):
    p = os.path.join(root, "synthetic", "synthetic_mechanism_ablation_summary.csv")
    df = pd.read_csv(p).set_index(["mask", "spectrum", "rank"])
    ROWS = [("indep", "flat", "Independent"), ("indep", "concentrated", "Independent"),
            ("group_32", "flat", "Groups, $G{=}32$"), ("group_8", "flat", "Groups, $G{=}8$"),
            ("group_2", "flat", "Groups, $G{=}2$"), ("group_2", "concentrated", "Groups, $G{=}2$"),
            ("colshare_0.10", "flat", "Shared cols, $c{=}0.10$"),
            ("colshare_0.25", "flat", "Shared cols, $c{=}0.25$"),
            ("colshare_0.40", "flat", "Shared cols, $c{=}0.40$")]
    body = []
    for mask, spec, lab in ROWS:
        r = df.loc[(mask, spec, "full")]
        d = r["delta_svdleft_rows"] * 100
        body.append(f"{lab} & {spec} & {r['rho_M']:.3f} & {r['rank_drop']:.1f} & {r['R10']:.3f} & "
                    + ("0.0\\%" if abs(d) < 0.05 else f"{d:+.1f}\\%") + r" \\")
    cap = (r"Synthetic factorial, full-rank dense matrices. Structure is recovered only when mask"
           r" correlation forces rank collapse, and then in the amount of the imposed violation."
           r" The $G{=}8$ row isolates correlation without rank consequence.")
    return _table(cap, "tab:synthetic", "llrrrr",
                  r"\textbf{Mask} & \textbf{Spectrum} & $\rho(M)$ & rank-drop & $R_{10}$ & $\Delta$", body)


def cr_a7(root):
    ROLE = {"target": "target", "shape_control": "shape control", "write_control": "write control"}
    body = []
    for m in ["llama-7b", "opt-125m"]:
        p = os.path.join(root, "oproj_mechanism", m, "wanda", "sp50", "summary_by_layer_type.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df = df.assign(_o=df["role"].map({"target": 0, "shape_control": 1, "write_control": 2})).sort_values("_o")
        if body:
            body.append(r"\midrule")
        for i, (_, r) in enumerate(df.iterrows()):
            body.append(f"{MNAME[m] if i == 0 else ''} & {r['layer_type'].replace('_', chr(92) + '_')} & "
                        f"{ROLE[r['role']]} & {r['xn_across_head_share']:.3f} & {r['rho_M']:.3f} & "
                        f"{_half_up(r['rank_drop_1em3'], 1):+} & {_p2(r['delta_svdleft_zero_rows'])} \\\\"
                        .replace("+0.0 ", "0.0 "))
    cap = (r"Layer-type comparison under whole-model pruning. All values are means over every block"
           r" of that layer type, so the target rows understate the anomaly (Table~\ref{tab:app_depth})."
           r" Head-block structure in the activation statistics is present only at the output"
           r" projection, and only there do the three properties fail.")
    return _table(cap, "tab:app_layertype", "lllrrrr",
                  r"\textbf{Model} & \textbf{Layer} & \textbf{Role} & across-head & $\rho(M)$ & rank-drop & $\Delta$",
                  body)


def cr_a8(root, model="llama-7b"):
    mech = _mech(root, model)
    t = mech[mech["role"] == "target"].sort_values("block_idx")
    nb = int(t["block_idx"].max()) + 1

    def row(label, s):
        dead = int(round((s["frac_heads_mostly_dead"] * s["n_heads"]).max()))
        rd = s["rank_drop_1em3"].mean()
        return (f"{label} & {s['xn_across_head_share'].mean():.3f} & {dead}/{int(s['n_heads'].iloc[0])} & "
                f"{s['rho_M'].mean():.3f} & {_half_up(rd, 0 if len(s) == 1 else 1):+} & "
                f"{_p2(round(s['delta_svdleft_zero_rows'].mean(), 4))} \\\\")
    body = [row(str(b), t[t["block_idx"] == b]) for b in (0, 1, 2)]
    body.append(row(f"3--{nb//2 - 1} (mean)", t[(t["block_idx"] >= 3) & (t["block_idx"] < nb // 2)]))
    body.append(row(f"{nb//2}--{nb-1} (mean)", t[t["block_idx"] >= nb // 2]))
    cap = (r"Depth profile of " + MNAME[model] + r" \texttt{o\_proj}. Dead heads, not head-block"
           r" heterogeneity, track the anomaly: the across-head share rises with depth while the"
           r" anomaly disappears after block 1.")
    return _table(cap, "tab:app_depth", "lrrrrr",
                  r"\textbf{Block} & across-head & dead heads & $\rho(M)$ & rank-drop & $\Delta$", body)


def cr_perlayer(fn):
    return lambda root: fn(PerLayer(os.path.join(root, "perlayer"), CR_MODELS))


CAMERA_READY = OrderedDict([
    ("table1", ("static      transformations evaluated", cr_table1, "")),
    ("table2", ("perlayer+wm main Delta across scales, whole-model in brackets", cr_table2, "perlayer wholemodel")),
    ("table3", ("wholemodel  generality across pruning methods (OPT-125M)", cr_table3, "wholemodel")),
    ("table4", ("perlayer    dense zero rows of QR / SVD two-sided", cr_perlayer(t_qrsvd), "perlayer")),
    ("table5", ("crosslayer  learned cross-layer rotations", cr_table5, "crosslayer_q")),
    ("table6", ("perlayer+wm out_proj/o_proj anomaly layers", cr_table6, "perlayer oproj_mechanism")),
    ("table7", ("perlayer    rank-drop, hi90, R10, rho(M) diagnostics", cr_perlayer(t_diagnostics), "perlayer")),
    ("a1",     ("perlayer    all metrics x thresholds", cr_a1, "perlayer")),
    ("a2",     ("HARDCODED   pre-pruning rotation (source scripts missing)", lambda root: t_rotation(None), "")),
    ("a3",     ("wholemodel  complete per-transform Delta x method", cr_a3, "wholemodel")),
    ("a4",     ("perlayer+wm out_proj rank / spectral diagnostics", cr_a4, "perlayer oproj_mechanism")),
    ("a5",     ("perlayer+wm out_proj eff-rank change and Delta", cr_a5, "perlayer oproj_mechanism")),
    ("a6",     ("synthetic   synthetic factorial, full-rank cells", cr_a6, "synthetic")),
    ("a7",     ("oproj       layer-type comparison (Wanda)", cr_a7, "oproj_mechanism")),
    ("a8",     ("oproj       LLaMA-7B o_proj depth profile", cr_a8, "oproj_mechanism")),
])


# ══════════════════════════════════════════════════════════════════════════════
TABLES = OrderedDict([
    ("paper",        "perlayer   all main-paper per-layer tables (= paper_tables.tex)"),
    ("transforms",   "static     Table 1, transformation families"),
    ("delta",        "perlayer   Table 2, main Delta(zero_rows) across model scales"),
    ("qrsvd",        "perlayer   dense zero rows of QR / SVD two-sided"),
    ("diagnostics",  "perlayer   rank drop, R10, rho(M) across model scales"),
    ("outproj",      "perlayer   out_proj/o_proj anomaly layers"),
    ("rotation",     "HARDCODED  pre-pruning rotation (source scripts missing)"),
    ("appendix",     "perlayer   appendix A1, all metrics x thresholds"),
    ("rebuttal",     "perlayer   sparsity ablation, SparseGPT comparison, out_proj vs SparseGPT (= rebuttal_tables.tex)"),
    ("methods",      "wholemodel Delta per transform x pruning method (--metric zero_rows)"),
    ("colsparsity",  "wholemodel same as methods with --metric zero_cols"),
    ("seeds",        "wholemodel calibration-seed variance + t-test"),
    ("ppl",          "wholemodel perplexity check + row-deletion (truncate_oproj)"),
    ("outproj_perrow","wholemodel per-row magnitude out_proj control (was rock.py)"),
    ("crosslayer",   "crosslayer cross-layer rotation Q"),
    ("oproj_mechanism","oproj    out_proj anomaly mechanism"),
    ("synthetic",    "synthetic  synthetic three-mechanism factorial"),
    ("all",          "every table whose inputs exist, from the default results/ layout"),
    ("camera_ready", "every camera-ready table below, in paper order (reads --results_root)"),
])
for _k, (_d, _, _) in CAMERA_READY.items():
    TABLES[_k] = "camera-ready " + _d

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--table", default="all", choices=list(TABLES))
parser.add_argument("--results_root", default=RES,
                    help="camera-ready tables: root holding perlayer/, wholemodel/, crosslayer_q/, "
                         "oproj_mechanism/, synthetic/ (default: results/)")
parser.add_argument("--results_dir", default=None,
                    help="default depends on --table: results/{perlayer,wholemodel,crosslayer_q,oproj_mechanism,synthetic}")
parser.add_argument("--schema", choices=["auto", "perlayer", "wholemodel"], default="auto")
parser.add_argument("--models", nargs="+", default=None)
parser.add_argument("--model", default="opt-125m", help="whole-model tables: model")
parser.add_argument("--sparsity", type=int, default=50, help="whole-model tables: sparsity %%")
parser.add_argument("--metric", default="zero_rows", choices=["zero_rows", "zero_cols", "block4x4"])
parser.add_argument("--sparsities", nargs="+", type=int, default=[50, 60, 70, 80])
parser.add_argument("--sgpt_sparsity", type=int, default=50)
parser.add_argument("--out", default="-", help="output .tex path ('-' = stdout)")
parser.add_argument("--list", action="store_true")
args = parser.parse_args()

DEFAULT_DIR = {"crosslayer": "crosslayer_q", "oproj_mechanism": "oproj_mechanism",
               "synthetic": "synthetic", "methods": "wholemodel", "colsparsity": "wholemodel",
               "seeds": "wholemodel", "ppl": "wholemodel", "outproj_perrow": "wholemodel"}


def build(table, results_dir=None):
    if table in CAMERA_READY:
        return CAMERA_READY[table][1](args.results_root)
    if table == "camera_ready":
        L = []
        for k, (_, fn, needs) in CAMERA_READY.items():
            missing = [d for d in needs.split() if not os.path.isdir(os.path.join(args.results_root, d))]
            if missing:
                print(f"  skip {k}: no {', '.join(missing)}/ under {args.results_root}", file=sys.stderr)
                continue
            L += [f"% ===== {k} =====", ""] + fn(args.results_root)
        return L
    rd = results_dir or os.path.join(RES, DEFAULT_DIR.get(table, "perlayer"))
    if table in ("transforms",):
        return t_transforms(None)
    if table == "rotation":   # hard-coded values, no CSV needed
        return t_rotation(None)
    if table in ("crosslayer", "oproj_mechanism", "synthetic"):
        return {"crosslayer": t_crosslayer, "oproj_mechanism": t_oproj_mechanism,
                "synthetic": t_synthetic}[table](rd)
    schema = args.schema if args.schema != "auto" else detect_schema(rd)
    if table in ("methods", "colsparsity", "seeds", "ppl", "outproj_perrow"):
        require(schema, "wholemodel", table)
        if table == "seeds":
            return t_seeds(rd)
        if table == "ppl":
            return t_ppl(rd)
        if table == "outproj_perrow":
            return t_outproj_perrow(rd, args.model, f"sp{args.sparsity}")
        return t_methods(rd, args.model, f"sp{args.sparsity}",
                         "zero_cols" if table == "colsparsity" else args.metric)
    require(schema, "perlayer", table)
    if table == "rebuttal":
        return rebuttal_tables(rd, args.models or ["opt-125m", "opt-1.3b"],
                               args.sparsities, args.sgpt_sparsity)
    models = args.models or [m for m in ALL_MODELS
                             if os.path.exists(os.path.join(rd, m, "layer_diagnostics.csv"))]
    P = PerLayer(rd, models)
    if table == "paper":
        # three blocks concatenated without a separator, as the original
        # generator wrote them (one write + two appends)
        return ["\n".join(t_transforms(P) + t_delta(P) + t_qrsvd(P) + t_diagnostics(P)
                          + t_outproj(P)) + "\n".join(t_rotation(P)) + "\n".join(t_appendix(P))]
    return {"delta": t_delta, "qrsvd": t_qrsvd, "diagnostics": t_diagnostics,
            "outproj": t_outproj, "rotation": t_rotation, "appendix": t_appendix}[table](P)


if __name__ == "__main__":
    if args.list:
        for k, v in TABLES.items():
            print(f"  {k:<16} {v}")
        sys.exit(0)
    if args.table == "all":
        L = []
        for t in ["paper", "rebuttal", "methods", "colsparsity", "seeds", "ppl",
                  "outproj_perrow", "crosslayer", "oproj_mechanism", "synthetic"]:
            try:
                L += [f"% ===== {t} =====", ""] + build(t, args.results_dir)
            except SystemExit as e:
                print(f"  skip {t}: {e}", file=sys.stderr)
    else:
        L = build(args.table, args.results_dir)
    text = "\n".join(L)
    if args.out == "-":
        print(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Wrote {args.out}  ({len(L)} lines)", file=sys.stderr)
