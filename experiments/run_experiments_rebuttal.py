"""
Per-layer protocol, rebuttal extension: sparsity sweep + per-layer SparseGPT.

Extends run_experiments.py (same protocol: each sampled layer pruned in
isolation, Wanda scalers from one dense-model forward pass) with:

  R-NEW1: Sparsity ablation. --sparsity takes several values, so one call
          sweeps Delta across 50/60/70/80%.
  R-NEW2: SparseGPT comparison. A SparseGPT variant (the official
          IST-DASLab/sparsegpt algorithm, Hessian-based sequential weight
          update, see pruning_transforms.pruning.SparseGPTWrapped) is added
          alongside Wanda and magnitude, under the same transform suite.
          The Hessian of each sampled layer is collected from the DENSE
          model, one layer at a time.

Claims substantiated: C1-C7 as in run_experiments.py, plus
  C8: Delta = 0.0% also holds for SparseGPT-pruned weights
  C9: Delta = 0.0% holds across sparsity 50-80%

Outputs under --results_dir/<model>/sp<NN>/: layer_diagnostics.csv,
transform_results.csv, sensitivity.csv, transform_failures.csv,
summary.json; plus rebuttal_summary.csv.

Usage:
  python experiments/run_experiments_rebuttal.py --models opt-125m opt-1.3b --sparsity 0.5 0.6 0.7 0.8
  python experiments/run_experiments_rebuttal.py --models opt-125m --sparsity 0.5 --skip_sparsegpt
"""
import os, sys, json, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning_transforms.config import (
    MODEL_REGISTRY, LAYER_PATTERNS, N_CALIB, SEQ_LEN, SPARSITY, SV_TOP_KS,
    N_BINS, MIN_DIM, MAX_C_OUT, RESULTS_ROOT, CALIB_CACHE,
    get_arch, is_eligible, sampled_blocks)
from pruning_transforms.metrics import (
    zero_row_frac, zero_col_frac, near_zero_row_frac, block_sparsity_frac,
    threshold_sensitivity, qr_theory_zero_rows, rank_at_thresholds,
    spectral_metrics, mask_corr)
from pruning_transforms.pruning import (
    collect_scalers, wanda_prune_gpu, magnitude_prune_global_gpu,
    magnitude_prune_perrow_gpu, magnitude_prune_hi_gpu, sparsegpt_prune_layer)
from pruning_transforms.transforms import ALL_TRANSFORMS
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["opt-125m", "opt-1.3b"],
                    choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--sparsity", type=float, nargs="+", default=[SPARSITY],
                    help="One or more sparsity ratios, e.g. --sparsity 0.5 0.6 0.7 0.8")
parser.add_argument("--skip_sparsegpt", action="store_true",
                    help="Skip the SparseGPT pruning variant (Wanda/magnitude only)")
parser.add_argument("--sparsegpt_blocksize", type=int, default=128,
                    help="Column blocksize for SparseGPT fasterprune (official default: 128)")
parser.add_argument("--sparsegpt_percdamp", type=float, default=0.01,
                    help="Hessian damping percentage for SparseGPT (official default: 0.01)")
parser.add_argument("--results_dir", default=str(RESULTS_ROOT / "perlayer"))
parser.add_argument("--calib_cache", default=str(CALIB_CACHE))
args = parser.parse_args()
RESULTS_DIR = args.results_dir


# ── Calibration data (C4 'en' stream, cached per model) ──────────────────────
def load_calib(tokenizer, model_key):
    os.makedirs(args.calib_cache, exist_ok=True)
    cache_path = os.path.join(args.calib_cache, f"{model_key}.pt")
    if os.path.exists(cache_path):
        print(f"  Loading cached calibration tokens from {cache_path}")
        return torch.load(cache_path)
    print("  Streaming C4 calibration data...")
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples, total = [], 0
    for row in data:
        enc = tokenizer(row["text"], return_tensors="pt",
                        max_length=SEQ_LEN, truncation=True)
        if enc["input_ids"].shape[1] == SEQ_LEN:
            samples.append(enc["input_ids"])
            total += 1
        if total >= N_CALIB:
            break
    tokens = torch.cat(samples, dim=0)
    torch.save(tokens, cache_path)
    return tokens


# ── Per-model run ─────────────────────────────────────────────────────────────
def run_model(model_key, sparsity):
    model_id = MODEL_REGISTRY[model_key]
    arch     = get_arch(model_key)
    sp_tag   = f"sp{int(round(sparsity * 100))}"
    out_dir  = os.path.join(RESULTS_DIR, model_key, sp_tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_key}  ({model_id})   sparsity={sparsity:.2f}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval().to(DEVICE)

    calib_tokens = load_calib(tokenizer, model_key)

    # Wanda scalers from ONE dense-model pass over all eligible layers,
    # mini-batches of 4 sequences.
    print("  Collecting activation statistics...")
    calib = calib_tokens.to(DEVICE)
    scalers = collect_scalers(model, [calib[i:i+4] for i in range(0, len(calib), 4)],
                              keep=lambda n: is_eligible(n, arch))
    print(f"  Got scalers for {len(scalers)} layers")

    # keep model on CPU for weight access but clear GPU activations
    model.cpu()
    torch.cuda.empty_cache()

    layer_diag_rows    = []
    transform_rows     = []
    sensitivity_rows   = []
    transform_failures = []

    # ── 10-bin depth sampling ────────────────────────────────────────────────
    # Normalised depth = bin_index / 9 in [0, 1]: the x-axis in figures, so
    # models of different depth are comparable on one plot.
    n_blocks = len(model.model.decoder.layers if arch == "opt" else model.model.layers)
    sample_indices = sampled_blocks(n_blocks)
    block_to_bin = {block_idx: bin_idx for bin_idx, block_idx in enumerate(sample_indices)}
    sample_strs = [f"layers.{i}." for i in sample_indices]
    print(f"  n_blocks={n_blocks}  sampled blocks={sample_indices}")
    print(f"  ({N_BINS} depth bins, one block per bin)")

    # SparseGPT needs forward passes through the live (dense) model to fill
    # its Hessian, so the model is GPU-resident for the loop when enabled.
    if not args.skip_sparsegpt:
        model.to(DEVICE)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear): continue
        if not is_eligible(name, arch): continue
        if name not in scalers: continue
        if not any(s in name for s in sample_strs): continue   # 10-bin filter

        W_tensor  = module.weight.data
        scaler    = scalers[name]
        C_out, C_in = W_tensor.shape

        if C_out < MIN_DIM or C_in < MIN_DIM or C_out > MAX_C_OUT: continue

        W_gpu    = W_tensor.float().to(DEVICE)
        xn_gpu   = scaler.float().to(DEVICE)          # (C_in,)

        # ── Pruning ──────────────────────────────────────────────────────────
        W_wanda_gpu  = wanda_prune_gpu(W_gpu, xn_gpu, sparsity)
        W_mag_gpu    = magnitude_prune_global_gpu(W_gpu, sparsity)
        W_mag_pr_gpu = magnitude_prune_perrow_gpu(W_gpu, sparsity)
        W_hi_gpu     = magnitude_prune_hi_gpu(W_gpu, 0.90)

        # SparseGPT (R-NEW2): fresh Hessian per sampled layer.
        W_sparsegpt_gpu = None
        if not args.skip_sparsegpt:
            try:
                W_sparsegpt_gpu = sparsegpt_prune_layer(
                    model, module, calib_tokens, sparsity, DEVICE,
                    blocksize=args.sparsegpt_blocksize,
                    percdamp=args.sparsegpt_percdamp)
            except Exception as e:
                transform_failures.append({
                    "model": model_key, "layer": name, "sparsity": sparsity,
                    "transform": "(pruning)", "pruning": "sparsegpt",
                    "C_out": C_out, "C_in": C_in,
                    "error_type": type(e).__name__, "error_msg": str(e)})
                print(f"    [SparseGPT FAILED on {name}: {type(e).__name__}: {e}]")

        W_np         = W_gpu.cpu().numpy()
        W_wanda      = W_wanda_gpu.cpu().numpy()
        W_mag_g      = W_mag_gpu.cpu().numpy()
        W_mag_pr     = W_mag_pr_gpu.cpu().numpy()

        # ── SVDs ─────────────────────────────────────────────────────────────
        S_dense  = torch.linalg.svdvals(W_gpu)
        S_wanda  = torch.linalg.svdvals(W_wanda_gpu)
        S_mag    = torch.linalg.svdvals(W_mag_gpu)
        S_hi     = torch.linalg.svdvals(W_hi_gpu)
        S_sgpt   = torch.linalg.svdvals(W_sparsegpt_gpu) if W_sparsegpt_gpu is not None else None

        # ── Rank at multiple thresholds, all anchored to dense sigma_max ─────
        S_dense_max = S_dense[0].item()
        ranks_d  = rank_at_thresholds(W_gpu,        S_dense_max)
        ranks_w  = rank_at_thresholds(W_wanda_gpu,  S_dense_max)
        ranks_m  = rank_at_thresholds(W_mag_gpu,    S_dense_max)
        ranks_mp = rank_at_thresholds(W_mag_pr_gpu, S_dense_max)
        ranks_hi = rank_at_thresholds(W_hi_gpu,     S_dense_max)
        ranks_sg = (rank_at_thresholds(W_sparsegpt_gpu, S_dense_max)
                    if W_sparsegpt_gpu is not None else None)
        # negative drop = spectral redistribution (real; never clamp)
        rank_drops_wanda = {t: ranks_d[t] - ranks_w[t] for t in ranks_d}
        rank_drops_mag   = {t: ranks_d[t] - ranks_m[t] for t in ranks_d}
        rank_drops_hi    = {t: ranks_d[t] - ranks_hi[t] for t in ranks_d}
        rank_drops_sgpt  = ({t: ranks_d[t] - ranks_sg[t] for t in ranks_d}
                            if ranks_sg is not None else None)

        spec_w   = spectral_metrics(S_dense, S_wanda)
        spec_m   = spectral_metrics(S_dense, S_mag)
        spec_hi  = spectral_metrics(S_dense, S_hi)
        spec_sg  = spectral_metrics(S_dense, S_sgpt) if S_sgpt is not None else None

        mc_w     = mask_corr(W_wanda)
        mc_m     = mask_corr(W_mag_g)
        mc_mp    = mask_corr(W_mag_pr)
        mc_sg    = (mask_corr(W_sparsegpt_gpu.cpu().numpy())
                    if W_sparsegpt_gpu is not None else None)
        # random mask control
        W_rand   = W_np.copy()
        W_rand[np.random.rand(*W_np.shape) < sparsity] = 0.0
        mc_r     = mask_corr(W_rand)

        layer_type = next((p for p in LAYER_PATTERNS[arch] if p in name), "other")
        block_idx  = next((int(p) for p in name.split(".") if p.isdigit()), 0)
        depth_bin  = block_to_bin.get(block_idx, 0)
        norm_depth = round(depth_bin / (N_BINS - 1), 4)   # 0.0 ... 1.0

        diag_row = {
            "model": model_key, "layer": name,
            "layer_type": layer_type,
            "sparsity": sparsity,
            "C_out": C_out, "C_in": C_in,
            "block_idx" : block_idx,
            "depth_bin" : depth_bin,
            "norm_depth": norm_depth,
            # C3: rank at multiple thresholds (negative = spectral redistribution)
            **{f"dense_{k}"       : v for k, v in ranks_d.items()},
            **{f"wanda_{k}"       : v for k, v in ranks_w.items()},
            **{f"mag_{k}"         : v for k, v in ranks_m.items()},
            **{f"mag_pr_{k}"      : v for k, v in ranks_mp.items()},
            **{f"hi90_{k}"        : v for k, v in ranks_hi.items()},
            **{f"drop_wanda_{k}"  : v for k, v in rank_drops_wanda.items()},
            **{f"drop_mag_{k}"    : v for k, v in rank_drops_mag.items()},
            **{f"drop_hi90_{k}"   : v for k, v in rank_drops_hi.items()},
            # C4: spectral
            "eff_rank_dense"   : spec_w["eff_rank_dense"],
            "eff_rank_wanda"   : spec_w["eff_rank_pruned"],
            "eff_rank_mag"     : spec_m["eff_rank_pruned"],
            "eff_rank_hi90"    : spec_hi["eff_rank_pruned"],
            **{f"wanda_top{k}_ratio": spec_w[f"top{k}_ratio"] for k in SV_TOP_KS},
            **{f"mag_top{k}_ratio"  : spec_m[f"top{k}_ratio"] for k in SV_TOP_KS},
            **{f"hi90_top{k}_ratio" : spec_hi[f"top{k}_ratio"] for k in SV_TOP_KS},
            # C5: mask correlation
            "mask_corr_wanda"  : mc_w,
            "mask_corr_mag_g"  : mc_m,
            "mask_corr_mag_pr" : mc_mp,
            "mask_corr_random" : mc_r,
            # C2: QR theoretical
            "qr_theory_zero_rows": qr_theory_zero_rows(C_out, C_in),
        }
        # C8: SparseGPT diagnostics, only present when computed
        if ranks_sg is not None:
            diag_row.update({f"sgpt_{k}": v for k, v in ranks_sg.items()})
            diag_row.update({f"drop_sgpt_{k}": v for k, v in rank_drops_sgpt.items()})
            diag_row["eff_rank_sgpt"] = spec_sg["eff_rank_pruned"]
            diag_row.update({f"sgpt_top{k}_ratio": spec_sg[f"top{k}_ratio"] for k in SV_TOP_KS})
            diag_row["mask_corr_sgpt"] = mc_sg
        layer_diag_rows.append(diag_row)

        # ── Transform loop ───────────────────────────────────────────────────
        pruning_variants = [
            (W_wanda_gpu,  "wanda"),
            (W_mag_gpu,    "magnitude_global"),
            (W_gpu,        "dense"),
        ]
        if W_sparsegpt_gpu is not None:
            pruning_variants.append((W_sparsegpt_gpu, "sparsegpt"))
        for tname, tfn in ALL_TRANSFORMS.items():
            for Wp_gpu, plabel in pruning_variants:
                try:
                    Wt = np.nan_to_num(tfn(Wp_gpu, xn_gpu))
                    transform_rows.append({
                        "model"         : model_key,
                        "layer"         : name,
                        "layer_type"    : layer_type,
                        "sparsity"      : sparsity,
                        "C_out"         : C_out,
                        "C_in"          : C_in,
                        "block_idx"     : block_idx,
                        "depth_bin"     : depth_bin,
                        "norm_depth"    : norm_depth,
                        "transform"     : tname,
                        "pruning"       : plabel,
                        "zero_rows"     : zero_row_frac(Wt),
                        "zero_cols"     : zero_col_frac(Wt),
                        "near_zero_rows": near_zero_row_frac(Wt),
                        "block4x4"      : block_sparsity_frac(Wt),
                    })
                    # threshold sensitivity for wanda and sparsegpt
                    if plabel in ("wanda", "sparsegpt"):
                        for zt, val in threshold_sensitivity(Wt).items():
                            sensitivity_rows.append({
                                "model"    : model_key, "layer": name,
                                "pruning"  : plabel,
                                "transform": tname, "threshold": zt,
                                "zero_rows": val,
                            })
                except Exception as e:
                    # Never swallow: a skipped transform is indistinguishable
                    # from a zero result downstream.
                    transform_failures.append({
                        "model": model_key, "layer": name,
                        "sparsity": sparsity, "transform": tname,
                        "pruning": plabel, "C_out": C_out, "C_in": C_in,
                        "error_type": type(e).__name__, "error_msg": str(e),
                    })
                    print(f"    [TRANSFORM FAILED] {tname} / {plabel} on "
                          f"{name} (C_out={C_out}, C_in={C_in}): "
                          f"{type(e).__name__}: {e}")

        drop_str = " | ".join(f"{k}:{v:+d}" for k, v in rank_drops_wanda.items())
        sgpt_str = (f"  sgpt_drop={rank_drops_sgpt['rank_1em3']:+d}"
                    if rank_drops_sgpt is not None else "")
        print(f"  ✓ {name:<45} rank_drops=[{drop_str}]  "
              f"top10={spec_w['top10_ratio']:.4f}  "
              f"mask_corr={mc_w:.4f}{sgpt_str}")

    if not args.skip_sparsegpt:
        model.cpu()
        torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────────
    df_diag  = pd.DataFrame(layer_diag_rows)
    df_trans = pd.DataFrame(transform_rows)
    df_sens  = pd.DataFrame(sensitivity_rows)
    df_fail  = pd.DataFrame(transform_failures, columns=[
        "model", "layer", "sparsity", "transform", "pruning", "C_out", "C_in",
        "error_type", "error_msg"])

    df_diag .to_csv(os.path.join(out_dir, "layer_diagnostics.csv"),  index=False)
    df_trans.to_csv(os.path.join(out_dir, "transform_results.csv"),  index=False)
    df_sens .to_csv(os.path.join(out_dir, "sensitivity.csv"),        index=False)
    df_fail .to_csv(os.path.join(out_dir, "transform_failures.csv"), index=False)

    if len(df_fail):
        print(f"\n  WARNING: {len(df_fail)} transform/layer/pruning combinations "
              f"failed -- see {out_dir}/transform_failures.csv")
        print(df_fail.groupby("transform").size().sort_values(ascending=False).to_string())

    # summary
    wanda_z = df_trans[df_trans["pruning"] == "wanda"].groupby("transform")["zero_rows"].mean()
    dense_z = df_trans[df_trans["pruning"] == "dense"].groupby("transform")["zero_rows"].mean()
    delta   = (wanda_z - dense_z).to_dict()

    # C8: Delta for SparseGPT, same construction as Wanda's
    has_sgpt = "sparsegpt" in df_trans["pruning"].unique() if len(df_trans) else False
    delta_sgpt = {}
    if has_sgpt:
        sgpt_z = df_trans[df_trans["pruning"] == "sparsegpt"].groupby("transform")["zero_rows"].mean()
        delta_sgpt = (sgpt_z - dense_z).to_dict()

    summary = {
        "model"               : model_key,
        "n_layers"            : len(layer_diag_rows),
        "sparsity"            : sparsity,
        "avg_rank_drop_wanda_1em3": float(df_diag["drop_wanda_rank_1em3"].mean()),
        "avg_rank_drop_wanda_1em6": float(df_diag["drop_wanda_rank_1em6"].mean()),
        "avg_rank_drop_mag_1em3"  : float(df_diag["drop_mag_rank_1em3"].mean()),
        "avg_rank_drop_hi90_1em3" : float(df_diag["drop_hi90_rank_1em3"].mean()),
        "avg_rank_drop_hi90_1em6" : float(df_diag["drop_hi90_rank_1em6"].mean()),
        "avg_eff_rank_dense"  : float(df_diag["eff_rank_dense"].mean()),
        "avg_eff_rank_wanda"  : float(df_diag["eff_rank_wanda"].mean()),
        "avg_top10_wanda"     : float(df_diag["wanda_top10_ratio"].mean()),
        "avg_top10_hi90"      : float(df_diag["hi90_top10_ratio"].mean()),
        "avg_mask_corr_wanda" : float(df_diag["mask_corr_wanda"].mean()),
        "avg_mask_corr_random": float(df_diag["mask_corr_random"].mean()),
        "max_delta_wanda"     : float(max(delta.values())) if delta else 0.0,
        "best_transform"      : max(delta, key=delta.get) if delta else "n/a",
        "all_delta_zero"      : all(abs(v) < 0.001 for v in delta.values()),
        "n_transform_failures": int(len(df_fail)),
        "delta_per_transform" : delta,
    }
    if has_sgpt:
        summary["avg_rank_drop_sgpt_1em3"] = float(df_diag["drop_sgpt_rank_1em3"].mean())
        summary["avg_rank_drop_sgpt_1em6"] = float(df_diag["drop_sgpt_rank_1em6"].mean())
        summary["avg_top10_sgpt"]          = float(df_diag["sgpt_top10_ratio"].mean())
        summary["avg_mask_corr_sgpt"]      = float(df_diag["mask_corr_sgpt"].mean())
        summary["max_delta_sgpt"]          = float(max(delta_sgpt.values())) if delta_sgpt else 0.0
        summary["best_transform_sgpt"]     = max(delta_sgpt, key=delta_sgpt.get) if delta_sgpt else "n/a"
        summary["all_delta_zero_sgpt"]     = all(abs(v) < 0.001 for v in delta_sgpt.values())
        summary["delta_per_transform_sgpt"] = delta_sgpt

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {out_dir}/")
    print(f"  rank_drop_wanda@1e-3 = {summary['avg_rank_drop_wanda_1em3']:.3f}")
    print(f"  rank_drop_wanda@1e-6 = {summary['avg_rank_drop_wanda_1em6']:.3f}")
    print(f"  rank_drop_hi90@1e-3  = {summary['avg_rank_drop_hi90_1em3']:.3f}")
    print(f"  top10_ratio      = {summary['avg_top10_wanda']:.4f}")
    print(f"  mask_corr_wanda  = {summary['avg_mask_corr_wanda']:.4f}")
    print(f"  max_delta_wanda  = {summary['max_delta_wanda']:+.4f}")
    print(f"  all_delta_zero   = {summary['all_delta_zero']}")
    if has_sgpt:
        print(f"  max_delta_sgpt   = {summary['max_delta_sgpt']:+.4f}")
        print(f"  all_delta_zero_sgpt = {summary['all_delta_zero_sgpt']}")
    return summary


# ── Cross-model summary ───────────────────────────────────────────────────────
def save_cross_model_summary(summaries):
    rows = [{
        "model"              : s["model"],
        "sparsity"           : s["sparsity"],
        "n_layers"           : s["n_layers"],
        "rank_drop_wanda_1em3": s["avg_rank_drop_wanda_1em3"],
        "rank_drop_wanda_1em6": s["avg_rank_drop_wanda_1em6"],
        "rank_drop_mag_1em3" : s["avg_rank_drop_mag_1em3"],
        "rank_drop_hi90_1em3": s["avg_rank_drop_hi90_1em3"],
        "rank_drop_hi90_1em6": s["avg_rank_drop_hi90_1em6"],
        "eff_rank_dense"     : s["avg_eff_rank_dense"],
        "eff_rank_wanda"     : s["avg_eff_rank_wanda"],
        "top10_ratio_wanda"  : s["avg_top10_wanda"],
        "top10_ratio_hi90"   : s["avg_top10_hi90"],
        "mask_corr_wanda"    : s["avg_mask_corr_wanda"],
        "mask_corr_random"   : s["avg_mask_corr_random"],
        "max_delta_wanda"    : s["max_delta_wanda"],
        "all_delta_zero"     : s["all_delta_zero"],
        # C8: SparseGPT columns, blank when not computed
        "rank_drop_sgpt_1em3": s.get("avg_rank_drop_sgpt_1em3"),
        "rank_drop_sgpt_1em6": s.get("avg_rank_drop_sgpt_1em6"),
        "top10_ratio_sgpt"   : s.get("avg_top10_sgpt"),
        "mask_corr_sgpt"     : s.get("avg_mask_corr_sgpt"),
        "max_delta_sgpt"     : s.get("max_delta_sgpt"),
        "all_delta_zero_sgpt": s.get("all_delta_zero_sgpt"),
    } for s in summaries]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "rebuttal_summary.csv"), index=False)
    print(f"\n{'='*70}\nCROSS-MODEL / CROSS-SPARSITY SUMMARY\n{'='*70}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    summaries, failed = [], []
    for model_key in args.models:
        for sparsity in args.sparsity:
            try:
                summaries.append(run_model(model_key, sparsity))
            except Exception as e:
                print(f"\nERROR on {model_key} @ sparsity={sparsity}: {e}")
                import traceback; traceback.print_exc()
                failed.append((model_key, sparsity))
    save_cross_model_summary(summaries)
    print("\nDone.")
    if failed:
        sys.exit(f"FAILED models: {failed}")
