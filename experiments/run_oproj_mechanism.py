"""
E4 (rebuttal, R1-W6/Q4a, R3-Q1): mechanistic account of the o_proj anomaly.

Reviewer questions:
  R1-Q4a: "Why is only this layer different — what is the mechanistic cause of
           its rank drop and elevated rho(M)?"
  R3-Q1 : "Why do out-projection layers behave differently? Is it related to
           layer depth, activation statistics, or matrix shape?"

Hypothesis under test (per-head concatenation):
  o_proj's input is the CONCATENATION of H attention heads' value outputs.
  Its input columns therefore come in H blocks of d_head, and the Wanda
  scoring statistic xn_j (activation norm of input feature j) is homogeneous
  WITHIN a head but heterogeneous ACROSS heads (weak heads produce uniformly
  small value outputs). Since Wanda prunes the bottom-50% of |W_ij|*xn_j PER
  ROW, strong across-head heterogeneity makes every row prune roughly the
  same weak-head columns => inter-row mask correlation => column-support
  shrinkage => rank drop. q/k/v_proj and FFN layers read the residual stream,
  whose feature norms lack this block structure, so the effect is absent.

What the script measures, for EVERY block (full depth, not 10-bin — this is
a targeted analysis of ~4 small matrices per block):

  For o_proj/out_proj and controls (q_proj = same shape/same depth,
  fc2 = write-side control):
    A. Activation-norm structure: variance decomposition of xn across vs
       within head-sized column groups (across_share in [0,1]; the hypothesis
       predicts across_share HIGH for o_proj, LOW for controls).
    B. Mask structure: rho(M) overall; per-head column kill rate; Spearman
       correlation between head activation norm and head zero rate (predicted
       strongly negative for o_proj only); fraction of fully-dead heads.
    C. Consequences: rank at 1e-3/1e-6 (anchored to dense sigma_max),
       rank-drop, R10 top-10 SV energy ratio, Delta(zero_rows) under SVD-left.
    D. Depth: block index recorded => depth trend of all of the above
       (R3's "is it depth?" answered by regressing A against depth).
    E. Method: run with --methods wanda sparsegpt => the Wanda-specific
       nature of the anomaly (SparseGPT's weight update breaks the pure
       per-row thresholding) is tested on the same layers.

  Shape is controlled by construction: q_proj has the identical (d, d) shape
  as o_proj at every depth, so any o_proj/q_proj difference cannot be shape.

Pruning is whole-model via the official locuslab/wanda code (vendored in
third_party/wanda). Activation scalers come from the DENSE model on the first
32 calibration sequences.

Usage:
  python experiments/run_oproj_mechanism.py \
      --models opt-125m --methods wanda sparsegpt --sparsity 0.5
  # llama-7b for the +10.5% anomaly layer itself:
  python experiments/run_oproj_mechanism.py \
      --models llama-7b --methods wanda --sparsity 0.5 --dtype float16
"""

import os, sys, json, argparse, warnings, types, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning_transforms.config import (
    MODEL_REGISTRY, ZERO_THRESH, ROW_THRESH, RESULTS_ROOT, DEFAULT_WANDA_REPO,
    get_arch)
from pruning_transforms.metrics import mask_corr, zero_row_frac
from pruning_transforms.pruning import collect_scalers, import_official_pruners
from pruning_transforms.transforms import tg_svd_left
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

TARGETS = {
    # layer_type -> role in the analysis
    "opt":   {"out_proj": "target", "q_proj": "shape_control", "fc2": "write_control"},
    "llama": {"o_proj":   "target", "q_proj": "shape_control", "down_proj": "write_control"},
}

RANK_THRESHOLDS = [(1e-3, "1em3"), (1e-6, "1em6")]   # subset of config.RANK_THRESHOLDS
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--wanda_repo", type=str, default=str(DEFAULT_WANDA_REPO))
parser.add_argument("--models", nargs="+", default=["opt-125m"],
                    choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--methods", nargs="+", default=["wanda"],
                    choices=["wanda", "sparsegpt", "magnitude"])
parser.add_argument("--sparsity", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--nsamples", type=int, default=128)
parser.add_argument("--dtype", type=str, default="float32",
                    choices=["float16", "float32"])
parser.add_argument("--results_dir", default=str(RESULTS_ROOT / "oproj_mechanism"))
args = parser.parse_args()
RESULTS_DIR = args.results_dir

def layer_role(name, arch):
    for pat, role in TARGETS[arch].items():
        if pat in name and "lm_head" not in name:
            return pat, role
    return None, None

# ── metrics ──────────────────────────────────────────────────────────────────
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12
    return float((rx * ry).sum() / denom)

def head_decomposition(xn, n_heads):
    """Variance decomposition of activation norms across head-sized column
    groups. Returns across_share in [0,1]: fraction of xn variance explained
    by head identity. High => rows prune the same weak-head columns."""
    d = len(xn); dh = d // n_heads
    xn = xn[:n_heads * dh].reshape(n_heads, dh)
    grand = xn.mean()
    across = ((xn.mean(axis=1) - grand) ** 2).mean()
    total  = ((xn - grand) ** 2).mean() + 1e-30
    return float(across / total), xn.mean(axis=1)   # share, per-head mean norm

def analyze_layer(name, pat, role, block_idx, W_dense, W_prun, xn, n_heads, rng):
    C_out, C_in = W_prun.shape
    Wd = W_dense.float().to(DEVICE)
    Wp = W_prun.float().to(DEVICE)
    Wp_np = Wp.cpu().numpy()

    # A. activation-norm head structure (head groups on the INPUT side)
    across_share, head_norms = head_decomposition(xn.numpy(), n_heads)

    # B. mask structure
    dh = C_in // n_heads
    mask_zero = (np.abs(Wp_np) < ZERO_THRESH)
    head_zero_rate = np.array([mask_zero[:, h*dh:(h+1)*dh].mean()
                               for h in range(n_heads)])
    dead_cols = (mask_zero.mean(axis=0) >= ROW_THRESH)
    head_dead_col_frac = np.array([dead_cols[h*dh:(h+1)*dh].mean()
                                   for h in range(n_heads)])
    rho = mask_corr(Wp_np, rng=rng)
    sp_corr = spearman(head_norms, head_zero_rate)   # predicted << 0 for target

    # C. spectral / rank consequences
    Sd = torch.linalg.svdvals(Wd); Sp = torch.linalg.svdvals(Wp)
    smax = Sd[0].item()
    ranks = {}
    for t, k in RANK_THRESHOLDS:
        ranks[f"rank_drop_{k}"] = int((Sd > smax*t).sum().item()
                                      - (Sp > smax*t).sum().item())
    R10 = float(Sp[:10].sum().item() / (Sd[:10].sum().item() + 1e-30))

    # Delta(zero_rows) under SVD-left (the anomaly's transform)
    delta_svdleft = zero_row_frac(tg_svd_left(Wp, None)) - zero_row_frac(tg_svd_left(Wd, None))

    return {
        "layer": name, "layer_type": pat, "role": role, "block_idx": block_idx,
        "C_out": C_out, "C_in": C_in, "n_heads": n_heads,
        # A
        "xn_across_head_share": across_share,
        "xn_head_norm_cv": float(head_norms.std() / (head_norms.mean() + 1e-30)),
        # B
        "rho_M": rho,
        "spearman_headnorm_zerorate": sp_corr,
        "head_zero_rate_max": float(head_zero_rate.max()),
        "head_zero_rate_min": float(head_zero_rate.min()),
        "frac_dead_cols": float(dead_cols.mean()),
        "frac_heads_mostly_dead": float((head_dead_col_frac > 0.5).mean()),
        # C
        **ranks, "R10": R10, "delta_svdleft_zero_rows": delta_svdleft,
    }, head_zero_rate, head_norms

def main():
    for model_key in args.models:
        arch     = get_arch(model_key)
        model_id = MODEL_REGISTRY[model_key]
        off = import_official_pruners(arch, args.wanda_repo)
        check_sparsity, get_loaders = off["check_sparsity"], off["get_loaders"]
        PRUNERS = {"wanda": off["prune_wanda"], "sparsegpt": off["prune_sparsegpt"],
                   "magnitude": off["prune_magnitude"]}
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        for method in args.methods:
            tag = os.path.join(RESULTS_DIR, model_key, method,
                               f"sp{int(round(args.sparsity*100))}")
            os.makedirs(tag, exist_ok=True)
            print(f"\n{'='*70}\n{model_key} | {method} | sparsity={args.sparsity}"
                  f"\n{'='*70}", flush=True)

            dtype = torch.float16 if args.dtype == "float16" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
                device_map=DEVICE)
            model.eval()
            model.seqlen = min(2048, getattr(model.config,
                               "max_position_embeddings", 2048))
            n_heads = model.config.num_attention_heads

            calib, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                   seqlen=model.seqlen, tokenizer=tokenizer)
            calib_batches = [b[0] for b in calib[:32]]

            print("  collecting dense-model activation scalers ...", flush=True)
            scalers = collect_scalers(model, calib_batches,
                                      keep=lambda n: layer_role(n, arch)[0] is not None)

            # dense snapshots of ALL target/control layers, every block
            dense_snapshots = {}
            for name, m in model.named_modules():
                if not isinstance(m, nn.Linear): continue
                pat, role = layer_role(name, arch)
                if pat is None or name not in scalers: continue
                dense_snapshots[name] = m.weight.data.detach().clone().cpu()
            print(f"  snapshot: {len(dense_snapshots)} layers (all blocks)",
                  flush=True)

            ns = types.SimpleNamespace(nsamples=args.nsamples, seed=args.seed,
                                       sparsity_ratio=args.sparsity,
                                       use_variant=False,
                                       cache_dir="llm_weights", model=model_id)
            print(f"  pruning whole model with official {method} ...", flush=True)
            PRUNERS[method](ns, model, tokenizer, torch.device(DEVICE))
            print(f"  achieved sparsity: {check_sparsity(model):.4f}", flush=True)

            rng = np.random.default_rng(args.seed)
            rows, per_head_rows = [], []
            todo = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, nn.Linear) and n in dense_snapshots]
            t0 = time.time()
            for li, (name, module) in enumerate(todo, 1):
                pat, role = layer_role(name, arch)
                block_idx = next((int(p) for p in name.split(".")
                                  if p.isdigit()), None)
                print(f"    [{li}/{len(todo)}] {name} ...", flush=True)
                row, head_zero_rate, head_norms = analyze_layer(
                    name, pat, role, block_idx, dense_snapshots[name],
                    module.weight.data, scalers[name], n_heads, rng)
                row.update({"model": model_key, "method": method,
                            "sparsity": args.sparsity, "seed": args.seed})
                rows.append(row)
                for h in range(n_heads):
                    per_head_rows.append({
                        "model": model_key, "method": method, "layer": name,
                        "layer_type": pat, "block_idx": block_idx, "head": h,
                        "head_mean_xn": float(head_norms[h]),
                        "head_zero_rate": float(head_zero_rate[h])})
            print(f"  analysis done in {time.time()-t0:.0f}s", flush=True)

            df  = pd.DataFrame(rows)
            dfh = pd.DataFrame(per_head_rows)
            df.to_csv(os.path.join(tag, "layer_mechanism.csv"), index=False)
            dfh.to_csv(os.path.join(tag, "per_head_detail.csv"), index=False)

            # summary: target vs controls, means over depth
            print("\n  Means by layer_type (target vs controls):")
            cols = ["xn_across_head_share", "rho_M",
                    "spearman_headnorm_zerorate", "frac_dead_cols",
                    "rank_drop_1em3", "R10", "delta_svdleft_zero_rows"]
            summ = df.groupby(["layer_type", "role"])[cols].mean().round(4)
            print(summ.to_string())
            summ.to_csv(os.path.join(tag, "summary_by_layer_type.csv"))

            # depth trend for the target layer
            tgt = df[df["role"] == "target"].sort_values("block_idx")
            trend = tgt[["block_idx", "xn_across_head_share", "rho_M",
                         "rank_drop_1em3", "delta_svdleft_zero_rows"]]
            trend.to_csv(os.path.join(tag, "target_depth_trend.csv"), index=False)
            print("\n  Target-layer depth trend written to target_depth_trend.csv")

            del model
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()