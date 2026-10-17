"""
Whole-model pruning + transform suite (rebuttal, all reviewers).

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

Setup:
  git clone https://github.com/locuslab/wanda            # official repo
  python run_wholemodel_experiments.py \
      --wanda_repo ./wanda --models opt-125m \
      --methods wanda sparsegpt --sparsity 0.5 --seeds 0 1 2 --eval_ppl

  # E3 (anomaly deletion):
  python run_wholemodel_experiments.py \
      --wanda_repo ./wanda --models llama-7b --methods wanda --sparsity 0.5 \
      --truncate_oproj 1 --eval_ppl

NOTE: the official repo splits LLaMA and OPT support (lib/prune.py vs
lib/prune_opt.py, main.py vs main_opt.py). The import shim below handles both;
if your checkout differs, adjust `import_official_pruners`.
"""

import os, sys, json, argparse, warnings, types, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import linalg as spla
from scipy.fft import dct
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
from scipy import stats as spstats
from sklearn.decomposition import PCA as skPCA
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_REGISTRY = {
    "opt-125m"  : "facebook/opt-125m",
    "opt-1.3b"  : "facebook/opt-1.3b",
    "opt-6.7b"  : "facebook/opt-6.7b",
    "opt-13b"   : "facebook/opt-13b",
    "llama-7b"  : "huggyllama/llama-7b",
    "llama-13b" : "huggyllama/llama-13b",
}
LAYER_PATTERNS = {
    "opt"  : ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"],
    "llama": ["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
}

ZERO_THRESH = 1e-6
ROW_THRESH  = 0.99
BLOCK_SIZE  = 4
SV_TOP_KS   = [5, 10, 20, 50]
RANK_THRESHOLDS = [(1e-3,'1em3'), (1e-4,'1em4'), (1e-5,'1em5'), (1e-6,'1em6')]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results_wholemodel"

parser = argparse.ArgumentParser()
parser.add_argument("--wanda_repo", type=str, required=True,
                    help="Path to a checkout of https://github.com/locuslab/wanda")
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
args = parser.parse_args()

# ── Official-repo import shim ─────────────────────────────────────────────────
def import_official_pruners(repo_path, arch):
    """Returns (prune_wanda, prune_sparsegpt, prune_magnitude, check_sparsity,
    eval_ppl, get_loaders) from the official locuslab/wanda checkout."""
    repo_path = os.path.abspath(repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from lib.data import get_loaders                      # shared
    if arch == "llama":
        from lib.prune import (prune_wanda, prune_sparsegpt,
                               prune_magnitude, check_sparsity)
        from lib.eval import eval_ppl
    else:
        # OPT pruners live in a parallel module in the official repo.
        # IMPORTANT: import pruners and eval SEPARATELY — a missing
        # lib/eval_opt.py must not silently fall the pruners back to the
        # LLaMA lib.prune (model.model.layers vs model.model.decoder.layers).
        from lib.prune_opt import (prune_wanda, prune_sparsegpt,
                                   prune_magnitude, check_sparsity)
        try:
            from lib.eval_opt import eval_ppl
        except ImportError:
            from lib.eval import eval_ppl
    return prune_wanda, prune_sparsegpt, prune_magnitude, check_sparsity, eval_ppl, get_loaders

def call_eval_ppl(eval_ppl_fn, ns, model, tokenizer, device):
    """Official eval_ppl signature changed across commits; try both."""
    try:
        return float(eval_ppl_fn(ns, model, tokenizer, device))
    except TypeError:
        return float(eval_ppl_fn(model, tokenizer, device))

# ── Helpers shared with the existing pipeline (same conventions) ─────────────
def get_arch(model_key):
    return "llama" if "llama" in model_key else "opt"

def is_eligible(name, arch):
    return (any(p in name for p in LAYER_PATTERNS[arch]) and "lm_head" not in name)

def zero_row_frac(W, zt=ZERO_THRESH, rt=ROW_THRESH):
    return float(((np.abs(W) < zt).mean(axis=1) >= rt).mean())

def zero_col_frac(W, zt=ZERO_THRESH, rt=ROW_THRESH):
    return float(((np.abs(W) < zt).mean(axis=0) >= rt).mean())

def near_zero_row_frac(W, thresh=0.01):
    norms = np.linalg.norm(W, axis=1)
    return float((norms < thresh * (norms.mean() + 1e-12)).mean())

def block_sparsity_frac(W, bs=BLOCK_SIZE, thresh=0.9):
    C_out, C_in = W.shape
    nr, nc = C_out // bs, C_in // bs
    if nr == 0 or nc == 0: return 0.0
    blocks = np.abs(W[:nr*bs, :nc*bs]).reshape(nr, bs, nc, bs)
    zero_frac = (blocks < ZERO_THRESH).mean(axis=(1, 3))
    return float((zero_frac >= thresh).mean())

def rank_at_thresholds(W, S_dense_max):
    S = torch.linalg.svdvals(W)
    return {f"rank_{k}": int((S > S_dense_max * t).sum().item())
            for t, k in RANK_THRESHOLDS}

def spectral_metrics(S_dense, S_pruned):
    def eff_rank(S):
        s = S / (S.sum() + 1e-12) + 1e-15
        return float(torch.exp(-(s * torch.log(s)).sum()).item())
    out = {"eff_rank_dense": eff_rank(S_dense), "eff_rank_pruned": eff_rank(S_pruned)}
    for k in SV_TOP_KS:
        kk = min(k, S_dense.shape[0], S_pruned.shape[0])
        out[f"top{k}_ratio"] = float(S_pruned[:kk].sum().item() /
                                     (S_dense[:kk].sum().item() + 1e-12))
    return out

def mask_corr_fast(W_pruned, max_rows=256):
    mask = (np.abs(W_pruned) < ZERO_THRESH).astype(np.float32)
    if mask.shape[0] > max_rows:
        idx = np.random.choice(mask.shape[0], max_rows, replace=False)
        mask = mask[idx]
    mask -= mask.mean(axis=1, keepdims=True)
    mask /= (np.linalg.norm(mask, axis=1, keepdims=True) + 1e-12)
    corr = mask @ mask.T
    off  = corr[np.triu_indices(len(corr), k=1)]
    return float(np.mean(np.abs(off)))

# ── Transform suite (identical definitions to run_experiments_rebuttal.py) ──
def tg_identity(Wp, xn):    return Wp.cpu().numpy()

def tg_svd_right(Wp, xn):
    _, _, Vt = torch.linalg.svd(Wp, full_matrices=False)
    return (Wp @ Vt.T).cpu().numpy()

def tg_svd_left(Wp, xn):
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
    return (torch.diag(S) @ Vt).cpu().numpy()

def tg_svd_twosided(Wp, xn):
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
    W2 = torch.zeros_like(Wp); k = S.shape[0]
    W2[:k, :k] = torch.diag(S)
    return W2.cpu().numpy()

def tg_svd_weighted(Wp, xn):
    U, S, Vt = torch.linalg.svd(Wp * xn.reshape(1,-1), full_matrices=False)
    return (torch.diag(S) @ Vt).cpu().numpy()

def tg_diagonal_scaling(Wp, xn):
    return (Wp * xn.reshape(1,-1)).cpu().numpy()

def tg_col_sort(Wp, xn):
    zero_frac = (Wp.abs() < ZERO_THRESH).float().mean(dim=0)
    return Wp[:, torch.argsort(zero_frac, descending=True)].cpu().numpy()

def tg_row_sort(Wp, xn):
    zero_frac = (Wp.abs() < ZERO_THRESH).float().mean(dim=1)
    return Wp[torch.argsort(zero_frac, descending=True)].cpu().numpy()

def tg_qr_left(Wp, xn):
    Q, R = torch.linalg.qr(Wp, mode='complete')
    return R.cpu().numpy()

def tg_hadamard_right(Wp, xn):
    n = Wp.shape[1]
    p = int(2**np.ceil(np.log2(n)))
    H_np = (spla.hadamard(p, dtype=np.float32) / np.sqrt(p)).astype(np.float32)
    H = torch.tensor(H_np, device=Wp.device)
    return (Wp @ H[:n,:n]).cpu().numpy()

def tg_dct_right(Wp, xn):
    n = Wp.shape[1]
    Q = torch.tensor(dct(np.eye(n, dtype=np.float32), axis=0, norm='ortho').T,
                     device=Wp.device)
    return (Wp @ Q).cpu().numpy()

def tg_pca_right(Wp, xn):
    xn_np = xn.cpu().numpy(); C_in = len(xn_np)
    rng   = np.random.default_rng(42)
    X     = rng.standard_normal((C_in + 10, C_in)).astype(np.float32) * xn_np
    pca   = skPCA(n_components=C_in); pca.fit(X)
    Q_gpu = torch.tensor(pca.components_.T, device=Wp.device)
    return (Wp @ Q_gpu).cpu().numpy()

def tg_lu_left(Wp, xn):
    _, _, U = spla.lu(Wp.cpu().numpy()); return U

def tg_cuthill_mckee(Wp, xn):
    W_np = Wp.cpu().numpy()
    adj  = csr_matrix((np.abs(W_np) > ZERO_THRESH).astype(np.float32))
    if adj.shape[0] == adj.shape[1]:
        return W_np[reverse_cuthill_mckee(adj)]
    return W_np

TRANSFORMS = {
    "Identity": tg_identity, "PCA_right": tg_pca_right, "SVD_right": tg_svd_right,
    "SVD_left": tg_svd_left, "SVD_twosided": tg_svd_twosided, "QR_left": tg_qr_left,
    "LU_left": tg_lu_left, "Hadamard_right": tg_hadamard_right, "DCT_right": tg_dct_right,
    "Diagonal_scaling": tg_diagonal_scaling, "Col_sort": tg_col_sort,
    "Row_sort": tg_row_sort, "SVD_weighted": tg_svd_weighted,
    "Cuthill_McKee": tg_cuthill_mckee,
}

# ── Wanda scaler collection on the dense model (for xn-dependent transforms) ─
class WrappedLayer:
    def __init__(self, layer):
        self.columns    = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros(self.columns, device=layer.weight.device)
        self.nsamples   = 0
    def add_batch(self, inp):
        if len(inp.shape) == 2: inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3: inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples   += tmp
        inp = inp.float()
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
    def get_scaler(self):
        return torch.sqrt(self.scaler_row)

def collect_scalers(model, calib_batches, arch):
    wrapped, handles = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear): continue
        if not is_eligible(name, arch): continue
        wl = WrappedLayer(module); wrapped[name] = wl
        def make_hook(w):
            def hook(m, inp, out): w.add_batch(inp[0].detach())
            return hook
        handles.append(module.register_forward_hook(make_hook(wl)))
    with torch.no_grad():
        for batch in calib_batches:
            model(batch.to(next(model.parameters()).device))
    for h in handles: h.remove()
    return {n: w.get_scaler().cpu() for n, w in wrapped.items()}

# ── Per-row (per-output) magnitude pruning ───────────────────────────────────
# R2's "one additional per-output method": per row, prune the smallest
# sparsity-fraction of weights by |W|. Same per-output comparison group as
# Wanda/SparseGPT but no activation statistics and no weight update, so it
# isolates the per-output grouping itself. Deterministic (no calibration):
# run with a single seed. No official implementation exists; this matches
# magnitude_prune_perrow from our original pipeline, applied whole-model.
def prune_magnitude_perrow(model, arch, sparsity):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and is_eligible(name, arch):
            W = m.weight.data
            n_prune = int(W.shape[1] * sparsity)
            if n_prune <= 0:
                continue
            _, idx = torch.topk(W.abs(), n_prune, dim=1,
                                largest=False, sorted=False)
            W.scatter_(1, idx, 0.0)

# ── 10-bin depth sampling (same as existing pipeline) ────────────────────────
def sampled_blocks(n_blocks, n_bins=10):
    edges = np.linspace(0, n_blocks, n_bins + 1)
    idxs = sorted(set(min(int((edges[b] + edges[b+1]) / 2), n_blocks - 1)
                      for b in range(n_bins)))
    return idxs

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
          f"({len(TRANSFORMS)} transforms x 2 variants each) ...", flush=True)
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
            "mask_corr": mask_corr_fast(W_prun.cpu().numpy()),
            "actual_sparsity": float((W_prun.abs() < ZERO_THRESH).float().mean().item()),
        })

        for tname, tfn in TRANSFORMS.items():
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
    pd.DataFrame(failures).to_csv(os.path.join(out_dir, "transform_failures.csv"), index=False)
    df_t.to_csv(os.path.join(out_dir, "transform_results.csv"), index=False)
    df_d.to_csv(os.path.join(out_dir, "layer_diagnostics.csv"), index=False)

    pruned_z = df_t[df_t.pruning == method].groupby("transform")[
        ["zero_rows", "zero_cols", "block4x4"]].mean()
    dense_z  = df_t[df_t.pruning == "dense"].groupby("transform")[
        ["zero_rows", "zero_cols", "block4x4"]].mean()
    delta    = (pruned_z - dense_z)
    delta.to_csv(os.path.join(out_dir, "delta_per_transform.csv"))
    return delta

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for model_key in args.models:
        arch     = get_arch(model_key)
        model_id = MODEL_REGISTRY[model_key]
        (prune_wanda, prune_sparsegpt, prune_magnitude,
         check_sparsity, eval_ppl, get_loaders) = import_official_pruners(
            args.wanda_repo, arch)
        PRUNERS = {"wanda": prune_wanda, "sparsegpt": prune_sparsegpt,
                   "magnitude": prune_magnitude}

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
                    blocks = sampled_blocks(n_blocks)
                    sample_strs = [f"layers.{i}." for i in blocks]
                    dense_snapshots = {}
                    for name, m in model.named_modules():
                        if (isinstance(m, nn.Linear) and is_eligible(name, arch)
                                and any(s in name for s in sample_strs)):
                            C_out, C_in = m.weight.shape
                            if C_out < 64 or C_in < 64 or C_out > 16384: continue
                            dense_snapshots[name] = m.weight.data.detach().clone().cpu()
                    print(f"  snapshot: {len(dense_snapshots)} sampled layers "
                          f"(blocks {blocks})")

                    calib, _ = get_loaders("c4", nsamples=args.nsamples, seed=seed,
                                           seqlen=model.seqlen, tokenizer=tokenizer)
                    calib_batches = [b[0] for b in calib[:8]]  # short pass for scalers
                    scalers = collect_scalers(model, calib_batches, arch)

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
    metric). Written to results_wholemodel/seed_variance_summary.csv."""
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
    sig = df[(df.p_value < 0.05) & (df.metric == "zero_rows")]
    if len(sig):
        print("Transforms with Delta(zero_rows) significantly != 0 (p<0.05):")
        print(sig.round(5).to_string(index=False))
    else:
        print("No transform shows Delta(zero_rows) significantly different from 0.")

if __name__ == "__main__":
    main()