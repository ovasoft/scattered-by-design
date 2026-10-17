"""
Render the Section 5 figures from per-layer CSVs (nothing is recomputed).

Reads:    <results_dir>/<model>/{layer_diagnostics,transform_results}.csv
          (per-layer schema, written by experiments/run_experiments.py)
Produces: <out_dir>/fig1_rank_drop.png   rank drop across thresholds
          <out_dir>/fig2_spectral.png    R10 spectral energy ratio vs depth
          <out_dir>/fig3_mask_corr.png   rho(M): Wanda vs random vs magnitude
          <out_dir>/fig4_diagnostic.png  rank drop / R10 vs max Delta
          <out_dir>/fig5_heatmap.png     mean Delta per transform x layer type

Usage:
  python analysis/generate_figures.py
  python analysis/generate_figures.py --results_dir results/perlayer --models opt-125m opt-1.3b opt-6.7b
"""

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
parser.add_argument("--results_dir", default=os.path.join(REPO, "results", "perlayer"))
parser.add_argument("--models", nargs="+",
                    default=["opt-125m","opt-1.3b","opt-6.7b","opt-13b",
                             "llama-7b","llama-13b"])
parser.add_argument("--out_dir", default=os.path.join(REPO, "figures"))
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
diag_dfs  = {}
trans_dfs = {}

for model in args.models:
    diag  = os.path.join(args.results_dir, model, "layer_diagnostics.csv")
    trans = os.path.join(args.results_dir, model, "transform_results.csv")
    if os.path.exists(diag):
        diag_dfs[model]  = pd.read_csv(diag)
        print(f"  Loaded {model}")
    else:
        print(f"  Missing {model}")
    if os.path.exists(trans):
        trans_dfs[model] = pd.read_csv(trans)

if not diag_dfs:
    print("No results found. Run experiments/run_experiments.py first.")
    raise SystemExit(1)

# ── Style ─────────────────────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "opt-125m":"OPT-125M","opt-1.3b":"OPT-1.3B","opt-6.7b":"OPT-6.7B",
    "opt-13b":"OPT-13B","llama-7b":"LLaMA-7B","llama-13b":"LLaMA-13B",
}
LAYER_MARKERS = {"fc1":"o","fc2":"s","q_proj":"^","k_proj":"v",
                 "v_proj":"D","out_proj":"*","up_proj":"P",
                 "down_proj":"X","gate_proj":"h","other":"."}
OPT_COLORS  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
LLAMA_COLORS= ["#9467bd","#8c564b"]
ALL_COLORS  = OPT_COLORS + LLAMA_COLORS

THRESHOLDS  = ["1e-3","1e-4","1e-5","1e-6"]
THRESH_COLS = ["drop_wanda_rank_1em3","drop_wanda_rank_1em4",
               "drop_wanda_rank_1em5","drop_wanda_rank_1em6"]

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.labelsize"   : 11,
    "axes.titlesize"   : 11,
    "legend.fontsize"  : 9,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "figure.dpi"       : 150,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Rank-drop across thresholds (supports Section 5.1)
# Shows sign reversal — spectral redistribution not rank collapse
# ══════════════════════════════════════════════════════════════════════════════
def fig1_rank_drop():
    completed = [m for m in args.models if m in diag_dfs
                 and all(c in diag_dfs[m].columns for c in THRESH_COLS)]
    if not completed:
        print("  Fig 1: no data"); return

    fig, axes = plt.subplots(1, len(completed),
                             figsize=(3.5 * len(completed), 3.8),
                             sharey=True)
    if len(completed) == 1:
        axes = [axes]

    for ax, model, color in zip(axes, completed, ALL_COLORS):
        df = diag_dfs[model]
        x  = np.arange(len(THRESHOLDS))

        # mean ± std across all layers
        means = [df[c].mean() for c in THRESH_COLS]
        stds  = [df[c].std()  for c in THRESH_COLS]

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.fill_between(x,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)
        ax.plot(x, means, "o-", color=color, lw=1.8, ms=5,
                label="Mean (all layers)")

        # hi90 control — mean
        hi90_cols = ["drop_hi90_rank_1em3","drop_hi90_rank_1em4",
                     "drop_hi90_rank_1em5","drop_hi90_rank_1em6"]
        if all(c in df.columns for c in hi90_cols):
            hi90_means = [df[c].mean() for c in hi90_cols]
            ax.plot(x, hi90_means, "s--", color="gray", lw=1.4, ms=4,
                    label="90% mag. (control)")

        ax.set_xticks(x)
        ax.set_xticklabels([f"$10^{{{t[1:]}}}$" for t in
                            ["-3","-4","-5","-6"]], fontsize=9)
        ax.set_xlabel("Threshold $\\tau$")
        ax.set_title(MODEL_DISPLAY.get(model, model))
        if ax is axes[0]:
            ax.set_ylabel("Mean rank-drop")
            ax.legend(loc="upper left", frameon=False)

    fig.suptitle("Fig. 1: Rank-drop across thresholds — sign reversal confirms\n"
                 "spectral redistribution, not rank collapse (§5.1)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "fig1_rank_drop.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Spectral energy ratio vs normalised depth (supports Section 5.2)
# R10 < 1.0 across all layers and depths — Wanda disperses energy
# ══════════════════════════════════════════════════════════════════════════════
def fig2_spectral():
    completed = [m for m in args.models if m in diag_dfs
                 and "wanda_top10_ratio" in diag_dfs[m].columns
                 and "norm_depth" in diag_dfs[m].columns]
    if not completed:
        print("  Fig 2: no data (need norm_depth column)"); return

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6,
               label="$R_{10} = 1.0$ (no change)")

    for model, color in zip(completed, ALL_COLORS):
        df = diag_dfs[model]
        # average R10 per depth bin and layer type
        grp = df.groupby("norm_depth")["wanda_top10_ratio"].mean()
        ax.plot(grp.index, grp.values, "o-", color=color, lw=1.6, ms=4,
                label=MODEL_DISPLAY.get(model, model), alpha=0.85)

    ax.set_xlabel("Normalised model depth")
    ax.set_ylabel("$R_{10}$ (top-10 SV energy ratio)")
    ax.set_ylim(0.85, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    #ax.legend(frameon=False, ncol=2, loc="lower right")
    #ax.set_title("Fig. 2: Spectral energy ratio $R_{10} < 1.0$ across all depths\n"
    #             "— Wanda disperses rather than concentrates energy (§5.2)",
    #             fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "fig2_spectral.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Mask correlation — Wanda vs random vs magnitude (Section 5.3)
# Shows near-independence of row pruning
# ══════════════════════════════════════════════════════════════════════════════
def fig3_mask_corr():
    completed = [m for m in args.models if m in diag_dfs
                 and "mask_corr_wanda" in diag_dfs[m].columns]
    if not completed:
        print("  Fig 3: no data"); return

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x     = np.arange(len(completed))
    width = 0.28
    labels = [MODEL_DISPLAY.get(m, m) for m in completed]

    w_means = [diag_dfs[m]["mask_corr_wanda"].mean()  for m in completed]
    r_means = [diag_dfs[m]["mask_corr_random"].mean() for m in completed]
    m_means = [diag_dfs[m]["mask_corr_mag_g"].mean()
               if "mask_corr_mag_g" in diag_dfs[m].columns
               else np.nan for m in completed]

    ax.bar(x - width, w_means, width, label="Wanda", color="#1f77b4", alpha=0.85)
    ax.bar(x,         r_means, width, label="Random (i.i.d.)", color="#aec7e8", alpha=0.85)
    if not all(np.isnan(m_means)):
        ax.bar(x + width, m_means, width, label="Magnitude", color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean inter-row mask correlation $\\rho(M)$")
    ax.legend(frameon=False)
    ax.set_title("Fig. 3: Wanda row masks are nearly uncorrelated (§5.3)\n"
                 "— only marginally above random i.i.d. baseline",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "fig3_mask_corr.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Diagnostic scatter — rank-drop vs Delta (Section 5.4)
# Validates that rank-drop predicts post-hoc compressibility
# ══════════════════════════════════════════════════════════════════════════════
def fig4_diagnostic():
    # Need both diag and transform results
    completed = [m for m in args.models
                 if m in diag_dfs and m in trans_dfs
                 and "drop_wanda_rank_1em3" in diag_dfs[m].columns]
    if not completed:
        print("  Fig 4: no data"); return

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    # compute per-layer max delta (wanda - dense) across all transforms
    for model, color in zip(completed, ALL_COLORS):
        df_d = diag_dfs[model]
        df_t = trans_dfs[model]

        w = df_t[df_t["pruning"]=="wanda"][["layer","transform","zero_rows"]].rename(
                columns={"zero_rows":"zr_w"})
        d = df_t[df_t["pruning"]=="dense"][["layer","transform","zero_rows"]].rename(
                columns={"zero_rows":"zr_d"})
        mg = w.merge(d, on=["layer","transform"])
        mg["delta"] = (mg["zr_w"] - mg["zr_d"]) * 100
        max_delta = mg.groupby("layer")["delta"].max().reset_index()
        max_delta.columns = ["layer","max_delta"]

        merged = df_d.merge(max_delta, on="layer", how="inner")
        if len(merged) == 0: continue

        rd3    = merged["drop_wanda_rank_1em3"].values
        r10    = merged["wanda_top10_ratio"].values
        delta  = merged["max_delta"].values
        lt     = merged["layer_type"].values if "layer_type" in merged.columns \
                 else ["other"] * len(merged)

        # Panel A: rank-drop vs delta
        for ltype in set(lt):
            mask = lt == ltype
            axes[0].scatter(rd3[mask], delta[mask],
                           marker=LAYER_MARKERS.get(ltype, "."),
                           color=color, alpha=0.6, s=25, label=None)

        # Panel B: R10 vs delta
        axes[1].scatter(r10, delta, color=color, alpha=0.6, s=25,
                       label=MODEL_DISPLAY.get(model, model))

    # Panel A formatting
    axes[0].axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes[0].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes[0].set_xlabel("Rank-drop @ $10^{-3}$ (Wanda)")
    axes[0].set_ylabel("Max $\\Delta$ across transforms (%)")
    axes[0].set_title("(a) Rank-drop vs structured sparsity gain")

    # Panel B formatting
    axes[1].axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.5,
                   label="$R_{10} = 1.0$")
    axes[1].set_xlabel("$R_{10}$ ratio (Wanda)")
    axes[1].set_ylabel("Max $\\Delta$ across transforms (%)")
    axes[1].set_title("(b) Spectral energy ratio vs structured sparsity gain")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle("Fig. 4: Diagnostic criteria — neither rank-drop nor $R_{10}$\n"
                 "predicts non-zero $\\Delta$ for Wanda-pruned matrices (§5.4)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "fig4_diagnostic.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Delta heatmap across transforms and layer types (bonus)
# Shows which transform x layer combinations ever produce non-zero delta
# ══════════════════════════════════════════════════════════════════════════════
def fig5_heatmap():
    completed = [m for m in args.models if m in trans_dfs]
    if not completed:
        print("  Fig 5: no data"); return

    all_trans = []
    for model in completed:
        df = trans_dfs[model]
        w  = df[df["pruning"]=="wanda"][["layer","transform","zero_rows","layer_type"]].rename(
                 columns={"zero_rows":"zr_w"})
        d  = df[df["pruning"]=="dense"][["layer","transform","zero_rows"]].rename(
                 columns={"zero_rows":"zr_d"})
        mg = w.merge(d, on=["layer","transform"])
        mg["delta"] = (mg["zr_w"] - mg["zr_d"]) * 100
        mg["model"] = model
        all_trans.append(mg)

    df_all = pd.concat(all_trans)

    # mean delta per (transform, layer_type)
    pivot = df_all.groupby(["transform","layer_type"])["delta"].mean().unstack(fill_value=0)

    # order transforms
    T_ORDER = ["PCA_right","Hadamard_right","DCT_right",
               "SVD_right","SVD_left","SVD_twosided","QR_left","LU_left",
               "Diagonal_scaling","SVD_weighted",
               "Col_sort","Row_sort","Cuthill_McKee","Identity"]
    pivot = pivot.reindex([t for t in T_ORDER if t in pivot.index])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label="Mean $\\Delta$ (%)", shrink=0.8)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Fig. 5: Mean $\\Delta$ per (transform, layer type) — green = near zero\n"
                 "averaged across all completed models", fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "fig5_heatmap.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")

# ── Run all ───────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig1_rank_drop()
fig2_spectral()
fig3_mask_corr()
fig4_diagnostic()
fig5_heatmap()
print(f"\nAll figures saved to {args.out_dir}/")
print("Include in paper with:")
print("  \\includegraphics[width=\\linewidth]{figures/fig1_rank_drop.png}")
