"""
Whole-model protocol: official Wanda/SparseGPT pruning + transform suite.

Motivation: the existing pipeline (run_experiments.py / run_experiments_rebuttal.py)
prunes each sampled layer FUNCTIONALLY IN ISOLATION, with Wanda scalers collected
from a single forward pass of the DENSE model. Official Wanda/SparseGPT prune the
whole model SEQUENTIALLY: calibration inputs to block i are computed from the
already-pruned blocks 0..i-1. This script closes that gap by delegating pruning
entirely to the official locuslab/wanda codebase, then running the paper's
transform suite on the resulting whole-model-pruned weights.

What this buys the rebuttal:
  - Removes the objection that Delta ~ 0 is an artifact of per-layer pruning
    (methodological robustness; supports R2/R4 generality).
  - E1 (R1-W1/Q1): --seeds N1 N2 ... reruns whole-model pruning with different
    calibration draws (official get_loaders seed), giving Delta variance over
    calibration randomness + a one-sample t-test against 0.
  - E3 (R1-Q4b, R3-Q2): --truncate_oproj IDX rank-truncates the given
    o_proj/out_proj after pruning (deleting the rows SVD-left zeroes) and
    --eval_ppl measures the perplexity impact, plus parameter-reduction math.
  - R3 reproducibility: pruning is the official implementation, so perplexity
    sanity-checks against published Wanda numbers (llama-7b @ 50%: 7.26 wikitext2).

The official code is vendored (patched) in third_party/wanda, the default
--wanda_repo, so no clone step is needed.

Usage:
  python experiments/run_wholemodel_experiments.py --models opt-125m \
      --methods wanda sparsegpt --sparsity 0.5 --seeds 0 1 2 --eval_ppl

  # E3 (anomaly deletion):
  python experiments/run_wholemodel_experiments.py --models llama-7b \
      --methods wanda --sparsity 0.5 --truncate_oproj 1 --eval_ppl

Scalers for the activation-aware transforms (PCA_right, SVD_weighted,
Diagonal_scaling) are collected from the DENSE model on the first 8
calibration sequences; pruning itself uses all --nsamples (128).
"""
import os, sys, json, argparse, warnings, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as spstats
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning_transforms.config import (
    MODEL_REGISTRY, ZERO_THRESH, ROW_THRESH, SV_TOP_KS, MIN_DIM, MAX_C_OUT,
    RESULTS_ROOT, DEFAULT_WANDA_REPO, get_arch, is_eligible, sampled_blocks)
from pruning_transforms.metrics import (
    zero_row_frac, zero_col_frac, near_zero_row_frac, block_sparsity_frac,
    rank_at_thresholds, spectral_metrics, mask_corr)
from pruning_transforms.pruning import (
    collect_scalers, prune_magnitude_perrow, import_official_pruners)
from pruning_transforms.transforms import ALL_TRANSFORMS
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--wanda_repo", type=str, default=str(DEFAULT_WANDA_REPO),
                    help="Path to the (patched) locuslab/wanda tree "
                         "(default: vendored third_party/wanda)")
parser.add_argument("--models", nargs="+", default=["opt-125m"],
                    choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--methods", nargs="+", default=["wanda"],
                    choices=["wanda", "sparsegpt", "magnitude",
                             "magnitude_perrow"])
parser.add_argument("--sparsity", type=float, nargs="+", default=[0.5])
parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="Calibration seeds passed to the official get_loaders. "
                         "Multiple seeds => E1 variance over calibration draws.")
parser.add_argument("--nsamples", type=int, default=128)
parser.add_argument("--eval_ppl", action="store_true",
                    help="Evaluate wikitext2 perplexity of the pruned model "
                         "(and after --truncate_oproj, if given).")
parser.add_argument("--truncate_oproj", type=int, default=None, metavar="BLOCK_IDX",
                    help="E3: after pruning, rank-truncate the o_proj/out_proj of "
                         "this block to its SVD-left non-zero-row rank and re-eval.")
parser.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "float32"])
parser.add_argument("--results_dir", default=str(RESULTS_ROOT / "wholemodel"))
args = parser.parse_args()
RESULTS_DIR = args.results_dir


def call_eval_ppl(eval_ppl_fn, ns, model, tokenizer, device):
    """Official eval_ppl signature changed across commits; try both."""
    try:
        return float(eval_ppl_fn(ns, model, tokenizer, device))
    except TypeError:
        return float(eval_ppl_fn(model, tokenizer, device))


# ── E3: SVD-left-guided rank truncation of one o_proj/out_proj ───────────────
def truncate_oproj(model, arch, block_idx):
    """Rank-truncate the attention output projection of `block_idx`:
    keep exactly the singular components whose SVD-left rows are non-zero
    (row-norm >= ZERO_THRESH under the paper's zero-row definition), i.e.
    delete the rows SVD-left zeroes. Returns bookkeeping dict."""
    target = "o_proj" if arch == "llama" else "out_proj"
    module = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and f"layers.{block_idx}." in name and target in name:
            module = m; mod_name = name; break
    if module is None:
        raise ValueError(f"no {target} found for block {block_idx}")

    W = module.weight.data.float()
    U, S, Vt = torch.linalg.svd(W.to(DEVICE), full_matrices=False)
    SVt = torch.diag(S) @ Vt
    row_zero = (SVt.abs() < ZERO_THRESH).float().mean(dim=1) >= ROW_THRESH
    k = int((~row_zero).sum().item())          # retained rank
    W_trunc = (U[:, :k] * S[:k]) @ Vt[:k, :]   # rank-k approx, same shape
    module.weight.data.copy_(W_trunc.to(module.weight.dtype).to(module.weight.device))

    C_out, C_in = W.shape
    dense_params = C_out * C_in
    fact_params  = k * (C_out + C_in)          # if stored factorized U_k(S_kV_k^T)
    return {
        "layer"          : mod_name,
        "C_out"          : C_out, "C_in": C_in,
        "rank_kept"      : k,
        "rows_deleted"   : C_out - k if C_out <= C_in else int(row_zero.sum().item()),
        "zero_row_frac"  : float(row_zero.float().mean().item()),
        "params_dense"   : dense_params,
        "params_factored": fact_params,
        "param_reduction_this_layer": 1.0 - fact_params / dense_params,
    }

# ── Per-run: transforms + diagnostics on sampled layers ──────────────────────
def analyze_model(model, arch, model_key, scalers, dense_snapshots, method,
                  sparsity, seed, out_dir):
    import time
    transform_rows, diag_rows, failures = [], [], []
    todo = [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n in dense_snapshots]
    n_todo = len(todo)
    t_start = time.time()
    print(f"  analyzing {n_todo} sampled layers "
          f"({len(ALL_TRANSFORMS)} transforms x 2 variants each) ...", flush=True)
    for li, (name, module) in enumerate(todo, 1):
        t_layer = time.time()

        W_dense = dense_snapshots[name].float().to(DEVICE)
        W_prun  = module.weight.data.float().to(DEVICE)
        xn      = scalers[name].float().to(DEVICE)
        C_out, C_in = W_prun.shape
        print(f"    [{li}/{n_todo}] {name} ({C_out}x{C_in}) ...", flush=True)

        S_dense = torch.linalg.svdvals(W_dense)
        S_prun  = torch.linalg.svdvals(W_prun)
        S_dense_max = S_dense[0].item()
        ranks_d = rank_at_thresholds(W_dense, S_dense_max)
        ranks_p = rank_at_thresholds(W_prun,  S_dense_max)
        spec    = spectral_metrics(S_dense, S_prun)
        diag_rows.append({
            "model": model_key, "layer": name, "method": method,
            "sparsity": sparsity, "seed": seed, "C_out": C_out, "C_in": C_in,
            **{f"dense_{k}": v for k, v in ranks_d.items()},
            **{f"pruned_{k}": v for k, v in ranks_p.items()},
            **{f"drop_{k}": ranks_d[k] - ranks_p[k] for k in ranks_d},
            "eff_rank_dense": spec["eff_rank_dense"],
            "eff_rank_pruned": spec["eff_rank_pruned"],
            **{f"top{k}_ratio": spec[f"top{k}_ratio"] for k in SV_TOP_KS},
            "mask_corr": mask_corr(W_prun.cpu().numpy()),
            "actual_sparsity": float((W_prun.abs() < ZERO_THRESH).float().mean().item()),
        })

        for tname, tfn in ALL_TRANSFORMS.items():
            for Wx, plabel in [(W_prun, method), (W_dense, "dense")]:
                try:
                    Wt = np.nan_to_num(tfn(Wx, xn))
                    transform_rows.append({
                        "model": model_key, "layer": name, "method": method,
                        "sparsity": sparsity, "seed": seed,
                        "transform": tname, "pruning": plabel,
                        "zero_rows": zero_row_frac(Wt),
                        "zero_cols": zero_col_frac(Wt),
                        "near_zero_rows": near_zero_row_frac(Wt),
                        "block4x4": block_sparsity_frac(Wt),
                    })
                except Exception as e:
                    failures.append({"model": model_key, "layer": name,
                                     "method": method, "sparsity": sparsity,
                                     "seed": seed, "transform": tname,
                                     "pruning": plabel,
                                     "error_type": type(e).__name__,
                                     "error_msg": str(e)})
                    print(f"      [TRANSFORM FAILED] {tname}/{plabel} on {name}: "
                          f"{type(e).__name__}: {e}", flush=True)

        dt = time.time() - t_layer
        elapsed = time.time() - t_start
        eta = elapsed / li * (n_todo - li)
        print(f"      done in {dt:.1f}s  (elapsed {elapsed/60:.1f}m, "
              f"ETA {eta/60:.1f}m)", flush=True)

    df_t = pd.DataFrame(transform_rows)
    df_d = pd.DataFrame(diag_rows)
    pd.DataFrame(failures, columns=[
        "model", "layer", "method", "sparsity", "seed", "transform", "pruning",
        "error_type", "error_msg"]).to_csv(os.path.join(out_dir, "transform_failures.csv"), index=False)
    df_t.to_csv(os.path.join(out_dir, "transform_results.csv"), index=False)
    df_d.to_csv(os.path.join(out_dir, "layer_diagnostics.csv"), index=False)

    pruned_z = df_t[df_t["pruning"] == method].groupby("transform")[
        ["zero_rows", "zero_cols", "block4x4"]].mean()
    dense_z  = df_t[df_t["pruning"] == "dense"].groupby("transform")[
        ["zero_rows", "zero_cols", "block4x4"]].mean()
    delta    = (pruned_z - dense_z)
    delta.to_csv(os.path.join(out_dir, "delta_per_transform.csv"))
    return delta

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for model_key in args.models:
        arch     = get_arch(model_key)
        model_id = MODEL_REGISTRY[model_key]
        off = import_official_pruners(arch, args.wanda_repo)
        check_sparsity, eval_ppl, get_loaders = (
            off["check_sparsity"], off["eval_ppl"], off["get_loaders"])
        PRUNERS = {"wanda": off["prune_wanda"], "sparsegpt": off["prune_sparsegpt"],
                   "magnitude": off["prune_magnitude"]}

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        for method in args.methods:
            for sparsity in args.sparsity:
                for seed in args.seeds:
                    tag = os.path.join(RESULTS_DIR, model_key, method,
                                       f"sp{int(round(sparsity*100))}", f"seed{seed}")
                    os.makedirs(tag, exist_ok=True)
                    print(f"\n{'='*70}\n{model_key} | {method} | sparsity={sparsity} "
                          f"| calib seed={seed}\n{'='*70}")

                    dtype = torch.float16 if args.dtype == "float16" else torch.float32
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
                        device_map=DEVICE)
                    model.eval()
                    model.seqlen = min(2048, getattr(model.config,
                                       "max_position_embeddings", 2048))

                    # args namespace for the official pruners
                    ns = types.SimpleNamespace(
                        nsamples=args.nsamples, seed=seed,
                        sparsity_ratio=sparsity, use_variant=False,
                        cache_dir="llm_weights", model=model_id)

                    # dense snapshots + scalers for sampled layers, BEFORE pruning
                    if arch == "opt":
                        n_blocks = len(model.model.decoder.layers)
                    else:
                        n_blocks = len(model.model.layers)
                    blocks = sampled_blocks(n_blocks)   # 10-bin depth sampling
                    sample_strs = [f"layers.{i}." for i in blocks]
                    dense_snapshots = {}
                    for name, m in model.named_modules():
                        if (isinstance(m, nn.Linear) and is_eligible(name, arch)
                                and any(s in name for s in sample_strs)):
                            C_out, C_in = m.weight.shape
                            if C_out < MIN_DIM or C_in < MIN_DIM or C_out > MAX_C_OUT: continue
                            dense_snapshots[name] = m.weight.data.detach().clone().cpu()
                    print(f"  snapshot: {len(dense_snapshots)} sampled layers "
                          f"(blocks {blocks})")

                    calib, _ = get_loaders("c4", nsamples=args.nsamples, seed=seed,
                                           seqlen=model.seqlen, tokenizer=tokenizer)
                    calib_batches = [b[0] for b in calib[:8]]  # short pass for scalers
                    scalers = collect_scalers(model, calib_batches,
                                              keep=lambda n: is_eligible(n, arch))

                    # ── whole-model pruning ──────────────────────────────────
                    if method == "magnitude_perrow":
                        print("  pruning whole model with per-row magnitude ...")
                        prune_magnitude_perrow(model, arch, sparsity)
                    else:
                        print(f"  pruning whole model with official {method} ...")
                        PRUNERS[method](ns, model, tokenizer, torch.device(DEVICE))
                    sp = check_sparsity(model)
                    print(f"  achieved sparsity: {sp:.4f}")

                    run_info = {"model": model_key, "method": method,
                                "sparsity_target": sparsity,
                                "sparsity_achieved": float(sp),
                                "seed": seed, "nsamples": args.nsamples}

                    if args.eval_ppl:
                        ppl = call_eval_ppl(eval_ppl, ns, model, tokenizer,
                                            torch.device(DEVICE))
                        run_info["ppl_wikitext2_pruned"] = ppl
                        print(f"  wikitext2 ppl (pruned): {ppl:.3f}")

                    # ── transforms + Delta on whole-model-pruned weights ─────
                    delta = analyze_model(model, arch, model_key, scalers,
                                          dense_snapshots, method, sparsity,
                                          seed, tag)
                    run_info["max_delta_zero_rows"] = float(delta["zero_rows"].max())
                    run_info["best_transform"] = str(delta["zero_rows"].idxmax())
                    print(delta.round(4).to_string())

                    # ── E3: anomaly rank truncation + re-eval ────────────────
                    if args.truncate_oproj is not None:
                        info = truncate_oproj(model, arch, args.truncate_oproj)
                        run_info["truncate_oproj"] = info
                        print(f"  truncated {info['layer']}: rank {info['rank_kept']}"
                              f"/{min(info['C_out'], info['C_in'])}, "
                              f"layer param reduction (factored) "
                              f"{info['param_reduction_this_layer']:.2%}")
                        if args.eval_ppl:
                            ppl2 = call_eval_ppl(eval_ppl, ns, model, tokenizer,
                                                 torch.device(DEVICE))
                            run_info["ppl_wikitext2_truncated"] = ppl2
                            print(f"  wikitext2 ppl (pruned+truncated): {ppl2:.3f}  "
                                  f"(delta {ppl2 - run_info.get('ppl_wikitext2_pruned', float('nan')):+.3f})")

                    with open(os.path.join(tag, "run_info.json"), "w") as f:
                        json.dump(run_info, f, indent=2)

                    del model
                    torch.cuda.empty_cache()

    # ── E1: aggregate Delta variance over calibration seeds ──────────────────
    aggregate_seed_variance()

def aggregate_seed_variance():
    """Collect delta_per_transform.csv across seeds; report mean, std, and a
    one-sample t-test of H0: Delta = 0 per (model, method, sparsity, transform,
    metric). Written to <results_dir>/seed_variance_summary.csv."""
    rows = []
    for root, _, files in os.walk(RESULTS_DIR):
        if "delta_per_transform.csv" in files and "seed" in os.path.basename(root):
            parts = root.split(os.sep)
            seed  = int(parts[-1].replace("seed", ""))
            sp    = parts[-2]; method = parts[-3]; model = parts[-4]
            df = pd.read_csv(os.path.join(root, "delta_per_transform.csv"))
            df["model"], df["method"], df["sparsity"], df["seed"] = model, method, sp, seed
            rows.append(df)
    if not rows:
        return
    alld = pd.concat(rows)
    out = []
    for (model, method, sp, tname), g in alld.groupby(
            ["model", "method", "sparsity", "transform"]):
        for metric in ["zero_rows", "zero_cols", "block4x4"]:
            vals = g[metric].values
            row = {"model": model, "method": method, "sparsity": sp,
                   "transform": tname, "metric": metric, "n_seeds": len(vals),
                   "delta_mean": float(np.mean(vals)),
                   "delta_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan}
            if len(vals) > 1 and np.std(vals, ddof=1) > 0:
                t, p = spstats.ttest_1samp(vals, 0.0)
                row["t_stat"], row["p_value"] = float(t), float(p)
            else:
                # zero variance: Delta is deterministic across calibration draws
                row["t_stat"], row["p_value"] = np.nan, np.nan
            out.append(row)
    df = pd.DataFrame(out)
    path = os.path.join(RESULTS_DIR, "seed_variance_summary.csv")
    df.to_csv(path, index=False)
    print(f"\nSeed-variance summary written to {path}")
    sig = df[(df["p_value"] < 0.05) & (df["metric"] == "zero_rows")]
    if len(sig):
        print("Transforms with Delta(zero_rows) significantly != 0 (p<0.05):")
        print(sig.round(5).to_string(index=False))
    else:
        print("No transform shows Delta(zero_rows) significantly different from 0.")

if __name__ == "__main__":
    main()