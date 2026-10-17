"""
E2 (rebuttal, R1-Q2 / R3-Q3 / R4): SliceGPT-style cross-layer rotation Q
applied to whole-model Wanda/SparseGPT-pruned weights.

Reviewer 1, Q2: "Can you extend the impossibility argument to, or empirically
test, a single SliceGPT-style cross-layer Q (from activation covariance)
applied to Wanda-pruned weights?"
Reviewer 3, Q3: "Would a simple cross-layer rotation change the result on a
small model?"

Method. SliceGPT's computational invariance rotates the residual stream by an
orthogonal Q derived from activation statistics; weights reading FROM the
stream become W' = W Q, weights writing TO the stream become W' = Q^T W.
We compute Q as the eigenbasis (descending eigenvalues) of the residual-stream
activation covariance accumulated at every transformer-block input over C4
calibration data — i.e., the data-dependent, cross-layer transform class the
paper's fixed per-layer suite excludes (Sec. 4 limitations; R1-W2).

Pipeline order (strictly post-hoc): (1) load model, (2) prune the WHOLE model
with the official Wanda/SparseGPT code, (3) collect residual-stream covariance
from the PRUNED model's own activations and build Q, (4) apply Q to the pruned
weights, (5) measure structured sparsity vs. the Q-transformed dense baseline.
The transform sees only what a post-hoc method would see: the pruned network.

Q modes:
  global     one Q for the whole model (covariance pooled over all block
             inputs) — the reviewer's "single cross-layer Q".
  per_block  Q_i from block i's input covariance (closer to actual SliceGPT).
  random     Haar-random orthogonal Q (control; continuity with the paper's
             random-orthogonal experiment, Sec. 4.6).

Read/write classification (which side Q applies to):
  OPT   : read = q_proj, k_proj, v_proj, fc1   -> W' = W Q(_i)
          write = out_proj, fc2                 -> W' = Q(_i)^T W
  LLaMA : read = q_proj, k_proj, v_proj, up_proj, gate_proj -> W' = W Q(_i)
          write = o_proj, down_proj             -> W' = Q(_i)^T W

Delta metric (paper convention): Delta = m(T(W_pruned)) - m(T(W_dense)) for
m in {zero_rows, zero_cols, near_zero_rows, near_zero_cols, block4x4}.
Because SliceGPT deletes SMALL-NORM slices rather than exactly-zero ones, the
operative metrics here are near_zero_rows/cols (row/col norm < 1% of mean):
Delta_near_zero asks whether pruning creates ADDITIONAL deletable slices in
the rotated basis beyond what the dense matrix already has.

Note on exactness: full SliceGPT invariance on OPT requires LayerNorm ->
RMSNorm absorption (Ashkboos et al., 2024, App. A). We do not modify the
model — we only analyze transformed weight matrices — so functional
invariance machinery is unnecessary for the sparsity question being asked.
State this in the rebuttal.

Pruning is whole-model via the official locuslab/wanda code (same as
run_wholemodel_experiments.py).

Usage:
  python run_crosslayer_q.py --wanda_repo ./wanda --models opt-125m \
      --methods wanda sparsegpt --sparsity 0.5 --seed 0 \
      --q_modes global per_block random
"""

import os, sys, json, argparse, warnings, types, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_REGISTRY = {
    "opt-125m"  : "facebook/opt-125m",
    "opt-1.3b"  : "facebook/opt-1.3b",
    "opt-6.7b"  : "facebook/opt-6.7b",
    "llama-7b"  : "huggyllama/llama-7b",
    "llama-13b" : "huggyllama/llama-13b",
}
READ_WRITE = {
    "opt": {"read": ["q_proj", "k_proj", "v_proj", "fc1"],
            "write": ["out_proj", "fc2"]},
    "llama": {"read": ["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"],
              "write": ["o_proj", "down_proj"]},
}

ZERO_THRESH = 1e-6
ROW_THRESH  = 0.99
BLOCK_SIZE  = 4
NEAR_ZERO   = 0.01     # slice norm < 1% of mean norm => deletable slice
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results_crosslayer_q"

parser = argparse.ArgumentParser()
parser.add_argument("--wanda_repo", type=str, required=True)
parser.add_argument("--models", nargs="+", default=["opt-125m"],
                    choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--methods", nargs="+", default=["wanda"],
                    choices=["wanda", "sparsegpt", "magnitude"])
parser.add_argument("--sparsity", type=float, nargs="+", default=[0.5])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--nsamples", type=int, default=128)
parser.add_argument("--cov_batches", type=int, default=32,
                    help="calibration sequences used for covariance collection")
parser.add_argument("--q_modes", nargs="+",
                    default=["global", "per_block", "random"],
                    choices=["global", "per_block", "random"])
parser.add_argument("--dtype", type=str, default="float32",
                    choices=["float16", "float32"])
args = parser.parse_args()

# ── official-repo import (same shim as run_wholemodel_experiments.py) ────────
def get_arch(model_key):
    return "llama" if "llama" in model_key else "opt"

def import_official_pruners(repo_path, arch):
    repo_path = os.path.abspath(repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from lib.data import get_loaders
    if arch == "llama":
        from lib.prune import (prune_wanda, prune_sparsegpt,
                               prune_magnitude, check_sparsity)
    else:
        from lib.prune_opt import (prune_wanda, prune_sparsegpt,
                                   prune_magnitude, check_sparsity)
    return prune_wanda, prune_sparsegpt, prune_magnitude, check_sparsity, get_loaders

# ── metrics (paper conventions) ──────────────────────────────────────────────
def zero_row_frac(W):
    return float(((np.abs(W) < ZERO_THRESH).mean(axis=1) >= ROW_THRESH).mean())

def zero_col_frac(W):
    return float(((np.abs(W) < ZERO_THRESH).mean(axis=0) >= ROW_THRESH).mean())

def near_zero_row_frac(W, thresh=NEAR_ZERO):
    norms = np.linalg.norm(W, axis=1)
    return float((norms < thresh * (norms.mean() + 1e-12)).mean())

def near_zero_col_frac(W, thresh=NEAR_ZERO):
    norms = np.linalg.norm(W, axis=0)
    return float((norms < thresh * (norms.mean() + 1e-12)).mean())

def block_sparsity_frac(W, bs=BLOCK_SIZE, thresh=0.9):
    nr, nc = W.shape[0] // bs, W.shape[1] // bs
    if nr == 0 or nc == 0: return 0.0
    blocks = np.abs(W[:nr*bs, :nc*bs]).reshape(nr, bs, nc, bs)
    return float(((blocks < ZERO_THRESH).mean(axis=(1, 3)) >= thresh).mean())

METRICS = {
    "zero_rows"     : zero_row_frac,
    "zero_cols"     : zero_col_frac,
    "near_zero_rows": near_zero_row_frac,
    "near_zero_cols": near_zero_col_frac,
    "block4x4"      : block_sparsity_frac,
}

# ── residual-stream covariance collection ────────────────────────────────────
def get_blocks(model, arch):
    return (model.model.decoder.layers if arch == "opt"
            else model.model.layers)

def collect_stream_covariance(model, arch, calib_batches):
    """Hook every transformer-block input; accumulate per-block covariance
    Sigma_i (d x d) and the pooled global Sigma. Returns (Sigma_global,
    {block_idx: Sigma_i}), both float64 numpy."""
    blocks = get_blocks(model, arch)
    d = model.config.hidden_size
    cov = {i: np.zeros((d, d), dtype=np.float64) for i in range(len(blocks))}
    cnt = {i: 0 for i in range(len(blocks))}
    handles = []

    def make_hook(i):
        def hook(module, inputs, output):
            x = inputs[0]                       # (B, L, d) residual stream
            x = x.detach().float().reshape(-1, x.shape[-1])
            cov[i] += (x.T @ x).cpu().double().numpy()
            cnt[i] += x.shape[0]
        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    dev = next(model.parameters()).device
    with torch.no_grad():
        for b in calib_batches:
            model(b.to(dev))
    for h in handles:
        h.remove()

    per_block = {i: cov[i] / max(cnt[i], 1) for i in cov}
    Sigma_global = sum(per_block.values()) / len(per_block)
    return Sigma_global, per_block

def eig_q(Sigma):
    """Orthogonal eigenbasis, eigenvalues descending (SliceGPT's PCA basis)."""
    evals, evecs = np.linalg.eigh(Sigma)
    order = np.argsort(evals)[::-1]
    return evecs[:, order].astype(np.float32)

def random_q(d, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q.astype(np.float32)

# ── layer classification & rotation ──────────────────────────────────────────
def classify(name, arch):
    for p in READ_WRITE[arch]["read"]:
        if p in name: return "read"
    for p in READ_WRITE[arch]["write"]:
        if p in name: return "write"
    return None

def block_index(name):
    return next((int(p) for p in name.split(".") if p.isdigit()), None)

def apply_q(W, side, Q):
    """W: torch (C_out, C_in) on DEVICE. Q: torch (d, d) on DEVICE.
    read:  W' = W Q      (rotates input/stream side, C_in == d)
    write: W' = Q^T W    (rotates output/stream side, C_out == d)"""
    if side == "read":
        if W.shape[1] != Q.shape[0]: return None
        return W @ Q
    else:
        if W.shape[0] != Q.shape[0]: return None
        return Q.T @ W

# ── 10-bin depth sampling (manuscript protocol) ──────────────────────────────
def sampled_blocks(n_blocks, n_bins=10):
    edges = np.linspace(0, n_blocks, n_bins + 1)
    return sorted(set(min(int((edges[b] + edges[b+1]) / 2), n_blocks - 1)
                      for b in range(n_bins)))

def is_eligible(name, arch):
    pats = READ_WRITE[arch]["read"] + READ_WRITE[arch]["write"]
    return any(p in name for p in pats) and "lm_head" not in name

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    for model_key in args.models:
        arch     = get_arch(model_key)
        model_id = MODEL_REGISTRY[model_key]
        (prune_wanda, prune_sparsegpt, prune_magnitude,
         check_sparsity, get_loaders) = import_official_pruners(args.wanda_repo, arch)
        PRUNERS = {"wanda": prune_wanda, "sparsegpt": prune_sparsegpt,
                   "magnitude": prune_magnitude}
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        for method in args.methods:
            for sparsity in args.sparsity:
                tag = os.path.join(RESULTS_DIR, model_key, method,
                                   f"sp{int(round(sparsity*100))}")
                os.makedirs(tag, exist_ok=True)
                print(f"\n{'='*70}\n{model_key} | {method} | sparsity={sparsity} "
                      f"| seed={args.seed}\n{'='*70}", flush=True)

                dtype = torch.float16 if args.dtype == "float16" else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
                    device_map=DEVICE)
                model.eval()
                model.seqlen = min(2048, getattr(model.config,
                                   "max_position_embeddings", 2048))
                d = model.config.hidden_size

                calib, _ = get_loaders("c4", nsamples=args.nsamples,
                                       seed=args.seed, seqlen=model.seqlen,
                                       tokenizer=tokenizer)
                calib_batches = [b[0] for b in calib[:args.cov_batches]]

                # 1. dense snapshots of sampled layers (manuscript protocol:
                #    10 depth bins, ALL linear layer types within each)
                n_blocks = len(get_blocks(model, arch))
                blocks   = sampled_blocks(n_blocks)
                sample_strs = [f"layers.{i}." for i in blocks]
                dense_snapshots = {}
                for name, m in model.named_modules():
                    if (isinstance(m, nn.Linear) and is_eligible(name, arch)
                            and any(s in name for s in sample_strs)):
                        C_out, C_in = m.weight.shape
                        if C_out < 64 or C_in < 64 or C_out > 16384: continue
                        dense_snapshots[name] = m.weight.data.detach().clone().cpu()
                print(f"  snapshot: {len(dense_snapshots)} sampled layers "
                      f"(blocks {blocks})", flush=True)

                # 2. whole-model pruning via official code (FIRST)
                ns = types.SimpleNamespace(nsamples=args.nsamples, seed=args.seed,
                                           sparsity_ratio=sparsity,
                                           use_variant=False,
                                           cache_dir="llm_weights", model=model_id)
                print(f"  pruning whole model with official {method} ...", flush=True)
                PRUNERS[method](ns, model, tokenizer, torch.device(DEVICE))
                sp = check_sparsity(model)
                print(f"  achieved sparsity: {sp:.4f}", flush=True)

                # 3. residual-stream covariance on the PRUNED model.
                #    The post-hoc transform only ever sees the pruned network:
                #    Q is derived from the pruned model's own activations,
                #    exactly as SliceGPT would if applied post-hoc.
                print("  collecting residual-stream covariance on pruned model "
                      f"({len(calib_batches)} sequences) ...", flush=True)
                Sigma_g, Sigma_blocks = collect_stream_covariance(
                    model, arch, calib_batches)

                Qs = {}
                if "global" in args.q_modes:
                    Qs["global"] = {None: torch.tensor(eig_q(Sigma_g), device=DEVICE)}
                if "per_block" in args.q_modes:
                    Qs["per_block"] = {i: torch.tensor(eig_q(S), device=DEVICE)
                                       for i, S in Sigma_blocks.items()}
                if "random" in args.q_modes:
                    Qs["random"] = {None: torch.tensor(random_q(d, args.seed),
                                                       device=DEVICE)}

                # 4. rotate + measure
                rows = []
                todo = [(n, m) for n, m in model.named_modules()
                        if isinstance(m, nn.Linear) and n in dense_snapshots]
                t0 = time.time()
                for li, (name, module) in enumerate(todo, 1):
                    side = classify(name, arch)
                    if side is None: continue
                    bidx = block_index(name)
                    W_dense = dense_snapshots[name].float().to(DEVICE)
                    W_prun  = module.weight.data.float().to(DEVICE)
                    print(f"    [{li}/{len(todo)}] {name} ({side}) ...", flush=True)

                    for qmode, qdict in Qs.items():
                        Q = qdict.get(bidx if qmode == "per_block" else None)
                        if Q is None: continue
                        for W, plabel in [(W_prun, method), (W_dense, "dense")]:
                            Wt = apply_q(W, side, Q)
                            if Wt is None:      # stream dim mismatch
                                continue
                            Wt = np.nan_to_num(Wt.cpu().numpy())
                            row = {"model": model_key, "layer": name,
                                   "block_idx": bidx, "side": side,
                                   "method": method, "sparsity": sparsity,
                                   "seed": args.seed, "q_mode": qmode,
                                   "pruning": plabel}
                            for mname, mfn in METRICS.items():
                                row[mname] = mfn(Wt)
                            rows.append(row)
                    elapsed = time.time() - t0
                    print(f"      done (elapsed {elapsed:.0f}s, "
                          f"ETA {elapsed/li*(len(todo)-li):.0f}s)", flush=True)

                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(tag, "crosslayer_q_results.csv"), index=False)

                # 5. Delta per (q_mode, metric)
                summary = {}
                for qmode in df.q_mode.unique():
                    sub = df[df.q_mode == qmode]
                    pruned = sub[sub.pruning == method].groupby("layer")[
                        list(METRICS)].mean()
                    dense  = sub[sub.pruning == "dense"].groupby("layer")[
                        list(METRICS)].mean()
                    delta  = (pruned - dense)
                    summary[qmode] = {
                        f"delta_{m}_mean": float(delta[m].mean())
                        for m in METRICS} | {
                        f"delta_{m}_max": float(delta[m].max())
                        for m in METRICS}
                    delta.to_csv(os.path.join(tag, f"delta_{qmode}.csv"))

                run_info = {"model": model_key, "method": method,
                            "sparsity_target": sparsity,
                            "sparsity_achieved": float(sp),
                            "seed": args.seed, "nsamples": args.nsamples,
                            "cov_batches": args.cov_batches,
                            "delta_summary": summary}
                with open(os.path.join(tag, "run_info.json"), "w") as f:
                    json.dump(run_info, f, indent=2)

                print("\n  Delta summary (pruned - dense), per q_mode:")
                for qmode, s in summary.items():
                    print(f"    {qmode:10s} "
                          + "  ".join(f"{m}:{s[f'delta_{m}_mean']:+.4f}"
                                      f"(max {s[f'delta_{m}_max']:+.4f})"
                                      for m in ["zero_rows", "near_zero_rows",
                                                "near_zero_cols"]))

                del model
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()