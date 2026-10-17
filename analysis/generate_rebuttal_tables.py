"""
generate_rebuttal_tables.py — Generates rebuttal-specific LaTeX tables:

  Table R1: Sparsity ablation. Delta (Wanda) across 50/60/70/80% sparsity,
            per model, per transform. Addresses R1-W2, R2-W3, R3-W4.

  Table R2: SparseGPT comparison. Delta (Wanda) vs Delta (SparseGPT) at a
            fixed sparsity, per model, per transform. Addresses R1-Q3,
            R2-W4, R3-W5, R4 (per-output generalization beyond Wanda).

  Table R3: out_proj anomaly under SparseGPT. Same diagnostic columns as
            the main paper's Table 4, but for the SparseGPT variant on the
            SAME layers Wanda flagged as anomalous, to make the contrast
            explicit. Addresses R3-Q2 ("why out_proj specifically").

Reads:  results/{model}/sp{N}/layer_diagnostics.csv
        results/{model}/sp{N}/transform_results.csv
        results/{model}/sp{N}/summary.json
(N = round(sparsity*100), matching run_experiments_rebuttal.py's output dirs)

Writes: rebuttal_tables.tex

Usage:
    python generate_rebuttal_tables.py
    python generate_rebuttal_tables.py --results_dir results \
        --models opt-125m opt-1.3b --sparsities 50 60 70 80
"""
import os, json, argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", default="results")
parser.add_argument("--models", nargs="+", default=["opt-125m", "opt-1.3b"])
parser.add_argument("--sparsities", nargs="+", type=int, default=[50, 60, 70, 80],
                     help="Sparsity levels as integer percentages, e.g. 50 60 70 80")
parser.add_argument("--sgpt_sparsity", type=int, default=50,
                     help="Sparsity level (as int percent) to use for the Wanda-vs-SparseGPT comparison table")
parser.add_argument("--out", default="rebuttal_tables.tex")
args = parser.parse_args()

PLACEHOLDER = r"\textbf{[X]}"

MNAME = {
    "opt-125m": "OPT-125M", "opt-1.3b": "OPT-1.3B", "opt-6.7b": "OPT-6.7B",
    "opt-13b": "OPT-13B", "llama-7b": "LLaMA-7B", "llama-13b": "LLaMA-13B",
}
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
    "PCA_right", "Hadamard_right", "DCT_right",
    "SVD_right", "SVD_left", "SVD_twosided", "QR_left", "LU_left",
    "Diagonal_scaling", "SVD_weighted",
    "Col_sort", "Row_sort", "Cuthill_McKee",
    "Identity",
]
TBREAKS = {"SVD_right", "Diagonal_scaling", "Col_sort", "Identity"}

# ── Load: nested by (model, sparsity) ───────────────────────────────────────
# Mirrors generate_tables2.py's loader, but walks the sp{N}/ subdirectory
# that run_experiments_rebuttal.py writes (vs. the flat results/{model}/
# layout the original pipeline used for the single-sparsity main-paper run).
diag  = {}   # (model, sp) -> DataFrame
trans = {}   # (model, sp) -> DataFrame
summ  = {}   # (model, sp) -> dict

for m in args.models:
    for sp in args.sparsities:
        key = (m, sp)
        sp_dir = os.path.join(args.results_dir, m, f"sp{sp}")
        dp = os.path.join(sp_dir, "layer_diagnostics.csv")
        tp = os.path.join(sp_dir, "transform_results.csv")
        jp = os.path.join(sp_dir, "summary.json")
        if os.path.exists(dp):
            diag[key] = pd.read_csv(dp)
            print(f"  Loaded {m} sp{sp} (diagnostics)")
        else:
            print(f"  Missing {m} sp{sp} (diagnostics) -- expected at {dp}")
        if os.path.exists(tp):
            trans[key] = pd.read_csv(tp)
        if os.path.exists(jp):
            with open(jp) as f:
                summ[key] = json.load(f)

def has_sgpt(model, sp):
    """True if this (model, sparsity) run includes a sparsegpt pruning variant."""
    key = (model, sp)
    if key not in trans: return False
    return "sparsegpt" in trans[key]["pruning"].unique()

# ── Delta helpers, parameterized by pruning variant ─────────────────────────
# Same construction as generate_tables2.py's delta_mean/layer_delta, but
# takes `pruning` as an argument so the identical logic serves both Wanda
# and SparseGPT without duplicated function bodies.
def delta_mean(model, sp, transform, pruning="wanda"):
    key = (model, sp)
    if key not in trans: return PLACEHOLDER
    t = trans[key]
    w = t.loc[(t["pruning"] == pruning) & (t["transform"] == transform), "zero_rows"]
    d = t.loc[(t["pruning"] == "dense")  & (t["transform"] == transform), "zero_rows"]
    if len(w) == 0 or len(d) == 0: return PLACEHOLDER
    v = (w.mean() - d.mean()) * 100
    return "0.0\\%" if abs(v) < 0.05 else f"{v:+.1f}\\%"

def layer_delta(model, sp, layer_name, transform, pruning="wanda"):
    key = (model, sp)
    if key not in trans: return PLACEHOLDER
    t = trans[key]
    w = t.loc[(t["layer"] == layer_name) & (t["transform"] == transform) &
              (t["pruning"] == pruning), "zero_rows"]
    d = t.loc[(t["layer"] == layer_name) & (t["transform"] == transform) &
              (t["pruning"] == "dense"), "zero_rows"]
    if len(w) == 0 or len(d) == 0: return PLACEHOLDER
    v = (w.iloc[0] - d.iloc[0]) * 100
    return f"{v:+.1f}\\%"

def diag_mean(model, sp, col, fmt="{:.3f}"):
    key = (model, sp)
    if key not in diag: return PLACEHOLDER
    df = diag[key]
    if col not in df.columns: return PLACEHOLDER
    v = df[col].mean()
    return PLACEHOLDER if pd.isna(v) else fmt.format(v)

def diag_meanstd(model, sp, col, fmt="{:.3f}"):
    key = (model, sp)
    if key not in diag: return PLACEHOLDER
    df = diag[key]
    if col not in df.columns: return PLACEHOLDER
    v, s = df[col].mean(), df[col].std()
    if pd.isna(v): return PLACEHOLDER
    return f"${fmt.format(v)} \\pm {fmt.format(s)}$"

L = []

# ── TABLE R1: Sparsity ablation (Wanda Delta across 50/60/70/80%) ───────────
# Addresses R1-W2, R2-W3, R3-W4: "is 50% the only practically interesting
# moderate sparsity, and where does the rank-preservation regime end?"
for m in args.models:
    n_sp = len(args.sparsities)
    cs   = "l" + "r" * n_sp
    sph  = " & ".join(r"\textbf{" + f"{sp}\\%" + "}" for sp in args.sparsities)
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
        vals = " & ".join(delta_mean(m, sp, tk, pruning="wanda") for sp in args.sparsities)
        L.append(TNAME.get(tk, tk) + " & " + vals + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# ── TABLE R2: Wanda vs. SparseGPT Delta at a fixed sparsity ─────────────────
# Addresses R1-Q3, R2-W4, R3-W5, R4: does the per-output-pruning argument
# generalize beyond Wanda specifically, or is SparseGPT's Hessian-based
# weight update structurally different?
sg = args.sgpt_sparsity
models_with_sgpt = [m for m in args.models if has_sgpt(m, sg)]
if models_with_sgpt:
    n_m = len(models_with_sgpt)
    cs2 = "l" + "rr" * n_m
    mh2 = " & ".join(
        r"\multicolumn{2}{c}{\textbf{" + MNAME.get(m, m) + "}}" for m in models_with_sgpt
    )
    sub2 = " & ".join(r"Wanda & SparseGPT" for _ in models_with_sgpt)
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
        for m in models_with_sgpt:
            cells.append(delta_mean(m, sg, tk, pruning="wanda"))
            cells.append(delta_mean(m, sg, tk, pruning="sparsegpt"))
        L.append(TNAME.get(tk, tk) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
else:
    L += [
        r"% No models have completed SparseGPT runs at "
        f"{sg}\\% sparsity -- Table R2 omitted.",
        "",
    ]

skipped_sgpt = [m for m in args.models if m not in models_with_sgpt]
if skipped_sgpt:
    print(f"  NOTE: no SparseGPT data at {sg}% sparsity for: {skipped_sgpt}"
          f" (Table R2 will only show: {models_with_sgpt})")

# ── TABLE R3: out_proj anomaly, Wanda vs SparseGPT on the SAME layers ───────
# Addresses R3-Q2 ("why does out_proj specifically show elevated mask
# correlation -- is it the layer, or Wanda's comparison group?"). Pulls the
# exact layers flagged anomalous under Wanda (drop_wanda_rank_1em3 > 10) and
# reports both methods' diagnostics side by side on those same layers.
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
found_r3 = False
for m in args.models:
    key = (m, sg)
    if key not in diag: continue
    df = diag[key]
    if "layer_type" not in df.columns or "drop_wanda_rank_1em3" not in df.columns:
        continue
    anomaly = df.loc[
        df["layer_type"].isin(["out_proj", "o_proj"]) & (df["drop_wanda_rank_1em3"] > 10)
    ].sort_values("drop_wanda_rank_1em3", ascending=False)
    for _, row in anomaly.iterrows():
        mn = MNAME.get(m, m)
        parts = row["layer"].split(".")
        try:
            idx = parts.index("layers")
            short = "layers." + parts[idx + 1] + r".out\_proj"
        except ValueError:
            short = parts[-1]
        rd_w = row.get("drop_wanda_rank_1em3", np.nan)
        rd_s = row.get("drop_sgpt_rank_1em3", np.nan) if "drop_sgpt_rank_1em3" in df.columns else np.nan
        mc_w = row.get("mask_corr_wanda", np.nan)
        mc_s = row.get("mask_corr_sgpt", np.nan) if "mask_corr_sgpt" in df.columns else np.nan
        rd_w_s = f"{rd_w:+.0f}" if not pd.isna(rd_w) else PLACEHOLDER
        rd_s_s = f"{rd_s:+.0f}" if not pd.isna(rd_s) else PLACEHOLDER
        mc_w_s = f"{mc_w:.3f}"  if not pd.isna(mc_w) else PLACEHOLDER
        mc_s_s = f"{mc_s:.3f}"  if not pd.isna(mc_s) else PLACEHOLDER
        L.append(f"{mn} & {short} & {rd_w_s} & {rd_s_s} & {mc_w_s} & {mc_s_s} \\\\")
        found_r3 = True
if not found_r3:
    L.append(r"\multicolumn{6}{c}{No anomaly layers in completed results.} \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# ── Write ─────────────────────────────────────────────────────────────────
with open(args.out, "w") as f:
    f.write("\n".join(L))

print(f"\nWrote {args.out}  ({len(L)} lines)")
print(f"  (model, sparsity) pairs loaded  : {sorted(diag.keys())}")
all_pairs = [(m, sp) for m in args.models for sp in args.sparsities]
print(f"  (model, sparsity) pairs pending : {[p for p in all_pairs if p not in diag]}")
print(f"  Models with SparseGPT @ {sg}%    : {models_with_sgpt}")