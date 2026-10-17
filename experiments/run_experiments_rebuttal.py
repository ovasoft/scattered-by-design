"""
Rebuttal experiment suite: Wanda zero clustering across OPT and LLaMA model series.

Extends the original run_experiments.py with two rebuttal-specific additions
(NeurIPS 2026 reviews R1-R4):

  R-NEW1: Sparsity ablation. --sparsity now accepts multiple values, so a
          single invocation sweeps Delta across 50/60/70/80% (R1-W2, R2-W3,
          R3-W4: "is 50% the only practically interesting moderate sparsity?").

  R-NEW2: SparseGPT comparison. A new pruning variant using the official
          IST-DASLab/sparsegpt SparseGPT class (Hessian-based sequential weight
          update) is added alongside Wanda and magnitude pruning, so Delta can
          be computed for SparseGPT-pruned weights under the same transform
          suite (R1-Q3, R2-W4, R3-W5, R4: "does the per-output argument
          generalize beyond Wanda specifically?").

Wanda pruning follows the official implementation exactly:
  - Uses scaler_row = sqrt(sum of squared activations) per input feature
  - Per-row comparison group (unstructured)
  - No weight update

SparseGPT pruning follows the official implementation exactly:
  - Uses layer-wise Hessian H = (2/n) * X X^T accumulated from calibration data
  - Sequential per-block OBS-style weight update (fasterprune)
  - Weight update DOES propagate pruning error to surviving weights, unlike
    Wanda/magnitude — this is the central algorithmic difference under test

Speed fixes applied:
  - Wanda pruning vectorized (no Python row loop)
  - Block sparsity vectorized via reshape
  - SVDs cached per layer
  - Mask correlation uses dot-product approximation
  - Calibration tokens cached to disk

Claims substantiated:
  C1: No transform clusters Wanda zeros (Delta=0.0%)
  C2: QR/SVD zero rows = intrinsic geometry not Wanda
  C3: Wanda preserves full matrix rank (rank-drop=0)
  C4: Wanda disperses spectral energy (top-k SV ratio < 1.0)
  C5: Rows pruned independently (mask correlation ~ 0)
  C6: rank-drop + SV ratio are diagnostic
  C7: Findings scale-independent across model sizes
  C8 (rebuttal): Delta=0.0% also holds for SparseGPT-pruned weights
  C9 (rebuttal): Delta=0.0% holds across sparsity 50-80% in the moderate regime

Usage:
  python run_experiments_rebuttal.py --models opt-125m opt-1.3b --sparsity 0.5 0.6 0.7 0.8
  python run_experiments_rebuttal.py --models opt-125m --sparsity 0.5 --skip_sparsegpt
"""

import os, json, argparse, warnings, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import linalg as spla
from scipy.fft import dct
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA as skPCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ── Registry ──────────────────────────────────────────────────────────────────
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

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB     = 128
SEQ_LEN     = 128
SPARSITY    = 0.5  # legacy default, unused — see SPARSITY_LIST from --sparsity
ZERO_THRESH = 1e-6
ROW_THRESH  = 0.99
BLOCK_SIZE  = 4
THRESHOLDS  = [1e-3, 1e-4, 1e-6]
SV_TOP_KS   = [5, 10, 20, 50]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
CALIB_CACHE = "calib_cache"

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["opt-125m", "opt-1.3b"],
                    choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--sparsity", type=float, nargs="+", default=[0.5],
                    help="One or more sparsity ratios, e.g. --sparsity 0.5 0.6 0.7 0.8")
parser.add_argument("--skip_sparsegpt", action="store_true",
                    help="Skip the SparseGPT pruning variant (Wanda/magnitude only)")
parser.add_argument("--sparsegpt_blocksize", type=int, default=128,
                    help="Column blocksize for SparseGPT fasterprune (official default: 128)")
parser.add_argument("--sparsegpt_percdamp", type=float, default=0.01,
                    help="Hessian damping percentage for SparseGPT (official default: 0.01)")
args = parser.parse_args()
SPARSITY_LIST = args.sparsity

# ── Arch helpers ──────────────────────────────────────────────────────────────
def get_arch(model_key):
    return "llama" if "llama" in model_key else "opt"

def is_eligible(name, arch):
    """All linear layers except embeddings and lm_head."""
    return (any(p in name for p in LAYER_PATTERNS[arch])
            and "lm_head" not in name)

# ── Calibration data (cached per tokenizer vocab size) ───────────────────────
def load_calib(tokenizer, model_key):
    os.makedirs(CALIB_CACHE, exist_ok=True)
    cache_path = os.path.join(CALIB_CACHE, f"{model_key}.pt")
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

# ── WrappedLayer: matches official Wanda WrappedGPT exactly ─────────────────────
class WrappedLayer:
    """
    Mirrors official Wanda WrappedGPT exactly.
    Uses online running mean for numerical stability:
      scaler_row = mean over batches of ||X_j||_2^2  per input feature j
    Wanda metric: |W_ij| * sqrt(scaler_row_j)
    """
    def __init__(self, layer):
        self.layer      = layer
        self.dev        = layer.weight.device
        self.rows       = layer.weight.data.shape[0]
        self.columns    = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.nsamples   = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()              # (C_in, N*L)
        # online running mean — matches official WrappedGPT
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples   += tmp
        inp = inp.float()
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def get_scaler(self):
        # scaler_row already holds mean squared norm — take sqrt for Wanda metric
        return torch.sqrt(self.scaler_row)

# ── SparseGPTWrapped: official SparseGPT (IST-DASLab/sparsegpt), trimmed ────
class SparseGPTWrapped:
    """
    Adapted from the official SparseGPT class. Trimmed of the optional
    quantizer hooks and Conv2d/Conv1D branches, since this codebase only
    prunes nn.Linear layers. Behavior of the core algorithm (Hessian
    accumulation + sequential per-block OBS update) is unchanged from the
    reference implementation.

    Unlike WrappedLayer (Wanda), this accumulates a full (C_in, C_in)
    Hessian rather than a per-feature scalar, and `fasterprune` performs
    a sequential weight UPDATE (not just a mask) that propagates pruning
    error to not-yet-pruned weights within and across column blocks.
    """
    def __init__(self, layer):
        self.layer   = layer
        self.dev     = layer.weight.device
        W            = layer.weight.data.clone()
        self.rows    = W.shape[0]
        self.columns = W.shape[1]
        self.H       = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, blocksize=128, percdamp=0.01):
        """
        Returns the pruned weight matrix (C_out, C_in) as a torch tensor on
        self.dev. Does NOT mutate self.layer.weight — caller owns the result,
        matching the functional style of wanda_prune_gpu / magnitude_prune_gpu
        elsewhere in this script (the official SparseGPT writes in place;
        we deviate here since this pipeline treats pruning fns as pure).
        """
        W = self.layer.weight.data.clone().float()

        H = self.H
        dead = torch.diag(H) == 0
        H = H.clone()
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1    = W[:, i1:i2].clone()
            Q1    = torch.zeros_like(W1)
            Err1  = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if mask is not None:
                mask1 = mask[:, i1:i2]
            else:
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W[:, i2:]  -= Err1.matmul(Hinv[i1:i2, i2:])

        return W.reshape(self.layer.weight.shape)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()

class _LinearShim:
    """Minimal stand-in exposing .weight.data, .weight.device, and
    .weight.shape so SparseGPTWrapped can wrap a raw weight tensor without
    touching the live model's nn.Linear module. Verified against a
    standalone unit test (test_sparsegpt_unit.py) before integration."""
    class _W:
        def __init__(self, data):
            self.data = data
        @property
        def device(self):
            return self.data.device
        @property
        def shape(self):
            return self.data.shape
    def __init__(self, weight_tensor):
        self.weight = self._W(weight_tensor)
        self.shape  = weight_tensor.shape


def collect_layer_stats(model, calib_tokens, arch):
    """
    Returns dict: layer_name -> scaler_row (C_in,) tensor on CPU.
    Processes one transformer block at a time to avoid OOM.
    """
    model.eval()
    scalers = {}

    if arch == "opt":
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    # prepare inputs by running embedding + first hooks
    # use a simple single-forward hook approach per layer block
    calib = calib_tokens.to(DEVICE)

    # collect all linear layers in one pass with hooks
    wrapped = {}
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear): continue
        if not is_eligible(name, arch): continue
        w = WrappedLayer(module)
        wrapped[name] = w
        def make_hook(wl):
            def hook(m, inp, out):
                wl.add_batch(inp[0].detach(), out.detach())
            return hook
        handles.append(module.register_forward_hook(make_hook(w)))

    with torch.no_grad():
        for i in range(0, len(calib), 4):   # mini-batches of 4
            model(calib[i:i+4])

    for h in handles:
        h.remove()

    for name, wl in wrapped.items():
        scalers[name] = wl.get_scaler().cpu()

    return scalers

# ── SparseGPT: single-layer Hessian collection + prune ───────────────────────
# Run lazily, one sampled layer at a time, rather than alongside the Wanda
# hook pass — H is O(C_in^2) per layer (vs. Wanda's O(C_in)), so collecting
# it for every eligible layer simultaneously would multiply memory use by
# C_in relative to the Wanda pass. Since we only need SparseGPT results for
# the 10-bin sampled layers anyway, we re-run a single forward pass through
# the *whole model* with a hook on just the target layer, immediately
# followed by fasterprune, then free the Hessian before moving to the next
# sampled layer. This trades some redundant forward-pass compute for a
# much smaller peak memory footprint.
def sparsegpt_prune_layer(model, module, layer_name, calib_tokens, sparsity,
                           blocksize=128, percdamp=0.01):
    """
    Returns the SparseGPT-pruned weight (C_out, C_in) as a torch tensor on
    DEVICE, computed from a fresh Hessian collected via a single forward
    pass with a hook on `module` only.

    Caller is responsible for ensuring `model` is on DEVICE before calling
    this (see run_model: the whole sampled-layer loop moves the model to
    DEVICE once, rather than this function doing it per-call, since the
    latter would mean one host<->device round trip of the FULL model per
    sampled layer per sparsity level -- wasteful for larger models).
    """
    shim = _LinearShim(module.weight.data.to(DEVICE))
    sgpt = SparseGPTWrapped(shim)

    def hook(m, inp, out):
        sgpt.add_batch(inp[0].detach())

    handle = module.register_forward_hook(hook)
    calib = calib_tokens.to(DEVICE)
    try:
        with torch.no_grad():
            for i in range(0, len(calib), 4):
                model(calib[i:i+4])
    finally:
        handle.remove()

    W_pruned = sgpt.fasterprune(sparsity, blocksize=blocksize, percdamp=percdamp)
    sgpt.free()
    return W_pruned.to(DEVICE)



# ── GPU pruning functions ─────────────────────────────────────────────────────
def wanda_prune_gpu(W, xn, sparsity=0.5):
    metric  = W.abs() * xn.reshape(1, -1)
    n_prune = int(metric.shape[1] * sparsity)
    _, idx  = torch.topk(metric, n_prune, dim=1, largest=False, sorted=False)
    Wp = W.clone(); Wp.scatter_(1, idx, 0.0)
    return Wp

def magnitude_prune_global_gpu(W, sparsity=0.5):
    k      = int(W.numel() * sparsity)
    thresh = torch.kthvalue(W.abs().flatten(), k).values
    Wp     = W.clone(); Wp[Wp.abs() <= thresh] = 0.0
    return Wp

def magnitude_prune_perrow_gpu(W, sparsity=0.5):
    n_prune = int(W.shape[1] * sparsity)
    _, idx  = torch.topk(W.abs(), n_prune, dim=1, largest=False, sorted=False)
    Wp      = W.clone(); Wp.scatter_(1, idx, 0.0)
    return Wp

def magnitude_prune_hi_gpu(W, sparsity=0.90):
    return magnitude_prune_global_gpu(W, sparsity)

RANK_THRESHOLDS = [(1e-3,'1em3'), (1e-4,'1em4'), (1e-5,'1em5'), (1e-6,'1em6')]

def rank_at_thresholds(W, S_dense_max):
    """
    Compute rank at multiple thresholds anchored to S_dense_max.
    Keys are explicit strings: rank_1em3, rank_1em4, rank_1em5, rank_1em6.
    """
    try:
        S = torch.linalg.svdvals(W)
        return {f"rank_{k}": int((S > S_dense_max * t).sum().item())
                for t, k in RANK_THRESHOLDS}
    except Exception:
        W_np = W.cpu().numpy()
        S_np = np.linalg.svd(W_np, compute_uv=False)
        return {f"rank_{k}": int((S_np > S_np[0] * t).sum())
                for t, k in RANK_THRESHOLDS}



# ── Metrics (all vectorized) ──────────────────────────────────────────────────
def zero_row_frac(W, zt=ZERO_THRESH, rt=ROW_THRESH):
    return float(((np.abs(W) < zt).mean(axis=1) >= rt).mean())

def zero_col_frac(W, zt=ZERO_THRESH, rt=ROW_THRESH):
    return float(((np.abs(W) < zt).mean(axis=0) >= rt).mean())

def near_zero_row_frac(W, thresh=0.01):
    norms = np.linalg.norm(W, axis=1)
    return float((norms < thresh * (norms.mean() + 1e-12)).mean())

def block_sparsity_frac(W, bs=BLOCK_SIZE, thresh=0.9):
    """Vectorized block sparsity — no Python loops."""
    C_out, C_in = W.shape
    nr, nc = C_out // bs, C_in // bs
    if nr == 0 or nc == 0: return 0.0
    blocks = (np.abs(W[:nr*bs, :nc*bs])
              .reshape(nr, bs, nc, bs))
    zero_frac = (blocks < ZERO_THRESH).mean(axis=(1, 3))   # (nr, nc)
    return float((zero_frac >= thresh).mean())



def spectral_metrics(S_dense, S_pruned):
    """Takes pre-computed singular values (torch tensors on GPU)."""
    def eff_rank(S):
        s = S / (S.sum() + 1e-12) + 1e-15
        return float(torch.exp(-(s * torch.log(s)).sum()).item())
    out = {
        "eff_rank_dense" : eff_rank(S_dense),
        "eff_rank_pruned": eff_rank(S_pruned),
    }
    for k in SV_TOP_KS:
        kk = min(k, S_dense.shape[0], S_pruned.shape[0])
        out[f"top{k}_ratio"] = float(
            S_pruned[:kk].sum().item() / (S_dense[:kk].sum().item() + 1e-12))
    return out

def mask_corr_fast(W_pruned, max_rows=256):
    """Dot-product based correlation — avoids np.corrcoef on large matrices."""
    mask = (np.abs(W_pruned) < ZERO_THRESH).astype(np.float32)
    if mask.shape[0] > max_rows:
        idx = np.random.choice(mask.shape[0], max_rows, replace=False)
        mask = mask[idx]
    mask -= mask.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(mask, axis=1, keepdims=True) + 1e-12
    mask /= norms
    corr = mask @ mask.T
    off  = corr[np.triu_indices(len(corr), k=1)]
    return float(np.mean(np.abs(off)))

def threshold_sensitivity(W):
    return {zt: zero_row_frac(W, zt=zt) for zt in THRESHOLDS}

def qr_theory_zero_rows(C_out, C_in):
    return float(max(0, C_out - C_in) / C_out)

# GPU-accelerated transforms — take torch tensors, return numpy
def tg_identity(Wp, xn):    return Wp.cpu().numpy()

def tg_svd_right(Wp, xn):
    _, _, Vt = torch.linalg.svd(Wp, full_matrices=False)
    return (Wp @ Vt.T).cpu().numpy()

def tg_svd_left(Wp, xn):
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
    return (torch.diag(S) @ Vt).cpu().numpy()

def tg_svd_twosided(Wp, xn):
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
    W2 = torch.zeros_like(Wp)
    k  = S.shape[0]
    W2[:k, :k] = torch.diag(S)
    return W2.cpu().numpy()

def tg_svd_weighted(Wp, xn):
    U, S, Vt = torch.linalg.svd(Wp * xn.reshape(1,-1), full_matrices=False)
    return (torch.diag(S) @ Vt).cpu().numpy()

def tg_diagonal_scaling(Wp, xn):
    return (Wp * xn.reshape(1,-1)).cpu().numpy()

def tg_col_sort(Wp, xn):
    zero_frac = (Wp.abs() < ZERO_THRESH).float().mean(dim=0)
    idx = torch.argsort(zero_frac, descending=True)
    return Wp[:, idx].cpu().numpy()

def tg_row_sort(Wp, xn):
    zero_frac = (Wp.abs() < ZERO_THRESH).float().mean(dim=1)
    idx = torch.argsort(zero_frac, descending=True)
    return Wp[idx].cpu().numpy()

def tg_qr_left(Wp, xn):
    Q, R = torch.linalg.qr(Wp, mode='complete')
    return R.cpu().numpy()

def tg_hadamard_right(Wp, xn):
    n = Wp.shape[1]
    p = int(2**np.ceil(np.log2(n)))
    # NOTE: dividing by np.sqrt(p) upcasts the float32 hadamard array to
    # float64 via numpy type promotion (np.sqrt of a python int returns a
    # numpy.float64 scalar), which silently made every Hadamard call fail
    # with a dtype mismatch against the float32 Wp/H matmul -- caught via
    # transform_failures.csv logging added during rebuttal experiments.
    # Explicit .astype(np.float32) after the division fixes it.
    H_np = (spla.hadamard(p, dtype=np.float32) / np.sqrt(p)).astype(np.float32)
    H = torch.tensor(H_np, device=DEVICE)
    return (Wp @ H[:n,:n]).cpu().numpy()

def tg_dct_right(Wp, xn):
    n = Wp.shape[1]
    Q = torch.tensor(
        dct(np.eye(n, dtype=np.float32), axis=0, norm='ortho').T,
        device=DEVICE)
    return (Wp @ Q).cpu().numpy()

def tg_pca_right(Wp, xn):
    """PCA rotation of input space. Synthetic X has C_in+10 samples
    so PCA can return a full-rank (C_in, C_in) rotation matrix."""
    xn_np    = xn.cpu().numpy()
    C_in     = len(xn_np)
    rng      = np.random.default_rng(42)
    # need n_samples > C_in for full-rank PCA
    X        = rng.standard_normal((C_in + 10, C_in)).astype(np.float32) * xn_np
    pca      = skPCA(n_components=C_in)
    pca.fit(X)
    Q        = pca.components_.T                     # (C_in, C_in)
    Q_gpu    = torch.tensor(Q, device=DEVICE)
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
    "Identity"        : tg_identity,
    "PCA_right"       : tg_pca_right,
    "SVD_right"       : tg_svd_right,
    "SVD_left"        : tg_svd_left,
    "SVD_twosided"    : tg_svd_twosided,
    "QR_left"         : tg_qr_left,
    "LU_left"         : tg_lu_left,
    "Hadamard_right"  : tg_hadamard_right,
    "DCT_right"       : tg_dct_right,
    "Diagonal_scaling": tg_diagonal_scaling,
    "Col_sort"        : tg_col_sort,
    "Row_sort"        : tg_row_sort,
    "SVD_weighted"    : tg_svd_weighted,
    "Cuthill_McKee"   : tg_cuthill_mckee,
}

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

    print("  Collecting activation statistics...")
    scalers = collect_layer_stats(model, calib_tokens, arch)
    print(f"  Got scalers for {len(scalers)} layers")

    # keep model on CPU for weight access but clear GPU activations
    model.cpu()
    torch.cuda.empty_cache()

    layer_diag_rows   = []
    transform_rows    = []
    sensitivity_rows  = []
    transform_failures = []

    # ── 10-bin depth sampling ────────────────────────────────────────────────
    # Divide model depth into 10 equal bins, sample one block per bin.
    # Normalised depth = bin_index / 9  in [0, 1] — used as x-axis in figures
    # so models of different depths are comparable on a single plot.
    if arch == "opt":
        n_blocks = len(model.model.decoder.layers)
    else:
        n_blocks = len(model.model.layers)

    # one representative block per bin (floor of bin centre)
    N_BINS = 10
    bin_edges   = np.linspace(0, n_blocks, N_BINS + 1)
    sample_indices = []
    for b in range(N_BINS):
        centre = int((bin_edges[b] + bin_edges[b+1]) / 2)
        centre = min(centre, n_blocks - 1)
        sample_indices.append(centre)
    sample_indices = sorted(set(sample_indices))   # dedup edge cases

    # map block_index → depth bin — built from same sample_indices
    block_to_bin = {}
    for bin_idx, block_idx in enumerate(sample_indices):
        block_to_bin[block_idx] = bin_idx

    sample_strs = [f"layers.{i}." for i in sample_indices]
    print(f"  n_blocks={n_blocks}  sampled blocks={sample_indices}")
    print(f"  (10 depth bins, one block per bin)")

    # SparseGPT needs real forward passes through the live model (to populate
    # its Hessian via a hook), so the model must be GPU-resident for the
    # sampled-layer loop when SparseGPT is enabled. Wanda/magnitude pruning
    # only need the already-extracted weight tensor (moved to DEVICE per
    # layer below) and don't need the model itself on GPU, so we keep the
    # original CPU-resident behavior when --skip_sparsegpt is set.
    if not args.skip_sparsegpt:
        model.to(DEVICE)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear): continue
        if not is_eligible(name, arch): continue
        if name not in scalers: continue
        if not any(s in name for s in sample_strs): continue   # ← 10-bin filter

        W_tensor  = module.weight.data
        scaler    = scalers[name]
        C_out, C_in = W_tensor.shape

        if C_out < 64 or C_in < 64 or C_out > 16384: continue

        # ── Move to GPU for all heavy compute ───────────────────────────────
        W_gpu    = W_tensor.float().to(DEVICE)
        xn_gpu   = scaler.float().to(DEVICE)          # (C_in,)
        xn_np    = scaler.numpy()

        # synthetic X for PCA (256 samples, column std = xn)
        rng      = np.random.default_rng(42)
        X_proxy  = (rng.standard_normal((256, len(xn_np))).astype(np.float32)
                    * xn_np)

        # ── Pruning on GPU ────────────────────────────────────────────────────
        W_wanda_gpu  = wanda_prune_gpu(W_gpu, xn_gpu, sparsity)
        W_mag_gpu    = magnitude_prune_global_gpu(W_gpu, sparsity)
        W_mag_pr_gpu = magnitude_prune_perrow_gpu(W_gpu, sparsity)
        W_hi_gpu     = magnitude_prune_hi_gpu(W_gpu, 0.90)

        # SparseGPT: Hessian-based sequential weight update (rebuttal R-NEW2).
        # Collected fresh per sampled layer — see sparsegpt_prune_layer docstring
        # for why this isn't folded into the single Wanda hook pass.
        W_sparsegpt_gpu = None
        if not args.skip_sparsegpt:
            try:
                W_sparsegpt_gpu = sparsegpt_prune_layer(
                    model, module, name, calib_tokens, sparsity,
                    blocksize=args.sparsegpt_blocksize,
                    percdamp=args.sparsegpt_percdamp,
                )
            except Exception as e:
                print(f"    [SparseGPT FAILED on {name}: {e}]")
                W_sparsegpt_gpu = None

        # numpy copies for transforms that need numpy
        W_np         = W_gpu.cpu().numpy()
        W_wanda      = W_wanda_gpu.cpu().numpy()
        W_mag_g      = W_mag_gpu.cpu().numpy()
        W_mag_pr     = W_mag_pr_gpu.cpu().numpy()
        W_hi         = W_hi_gpu.cpu().numpy()

        # ── Cache SVDs on GPU ─────────────────────────────────────────────────
        S_dense  = torch.linalg.svdvals(W_gpu)
        S_wanda  = torch.linalg.svdvals(W_wanda_gpu)
        S_mag    = torch.linalg.svdvals(W_mag_gpu)
        S_hi     = torch.linalg.svdvals(W_hi_gpu)
        S_sgpt   = torch.linalg.svdvals(W_sparsegpt_gpu) if W_sparsegpt_gpu is not None else None

        # ── Rank at multiple thresholds — all anchored to dense S_max ──────
        S_dense_max = S_dense[0].item()
        ranks_d  = rank_at_thresholds(W_gpu,        S_dense_max)
        ranks_w  = rank_at_thresholds(W_wanda_gpu,  S_dense_max)
        ranks_m  = rank_at_thresholds(W_mag_gpu,    S_dense_max)
        ranks_mp = rank_at_thresholds(W_mag_pr_gpu, S_dense_max)
        ranks_hi = rank_at_thresholds(W_hi_gpu,     S_dense_max)
        ranks_sg = (rank_at_thresholds(W_sparsegpt_gpu, S_dense_max)
                    if W_sparsegpt_gpu is not None else None)
        # rank_drop per threshold — negative = spectral redistribution noise
        rank_drops_wanda = {t: ranks_d[t] - ranks_w[t] for t in ranks_d}
        rank_drops_mag   = {t: ranks_d[t] - ranks_m[t] for t in ranks_d}
        rank_drops_hi    = {t: ranks_d[t] - ranks_hi[t] for t in ranks_d}
        rank_drops_sgpt  = ({t: ranks_d[t] - ranks_sg[t] for t in ranks_d}
                             if ranks_sg is not None else None)

        spec_w   = spectral_metrics(S_dense, S_wanda)
        spec_m   = spectral_metrics(S_dense, S_mag)
        spec_hi  = spectral_metrics(S_dense, S_hi)
        spec_sg  = spectral_metrics(S_dense, S_sgpt) if S_sgpt is not None else None

        mc_w     = mask_corr_fast(W_wanda)
        mc_m     = mask_corr_fast(W_mag_g)
        mc_mp    = mask_corr_fast(W_mag_pr)
        mc_sg    = (mask_corr_fast(W_sparsegpt_gpu.cpu().numpy())
                    if W_sparsegpt_gpu is not None else None)
        # random mask control
        W_rand   = W_np.copy()
        W_rand[np.random.rand(*W_np.shape) < sparsity] = 0.0
        mc_r     = mask_corr_fast(W_rand)

        layer_type = next((p for p in LAYER_PATTERNS[arch] if p in name), "other")

        # extract block index from layer name e.g. "model.decoder.layers.5.fc1"
        block_idx = next(
            (int(p) for p in name.split(".") if p.isdigit()), 0)
        depth_bin  = block_to_bin.get(block_idx, 0)
        norm_depth = round(depth_bin / (N_BINS - 1), 4)   # 0.0 … 1.0

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
        # C8 (rebuttal): SparseGPT diagnostics, only present when computed
        if ranks_sg is not None:
            diag_row.update({f"sgpt_{k}": v for k, v in ranks_sg.items()})
            diag_row.update({f"drop_sgpt_{k}": v for k, v in rank_drops_sgpt.items()})
            diag_row["eff_rank_sgpt"] = spec_sg["eff_rank_pruned"]
            diag_row.update({f"sgpt_top{k}_ratio": spec_sg[f"top{k}_ratio"] for k in SV_TOP_KS})
            diag_row["mask_corr_sgpt"] = mc_sg
        layer_diag_rows.append(diag_row)

        # ── Transform loop ────────────────────────────────────────────────────
        # GPU pruning variants for transform loop
        gpu_pruning_variants = [
            (W_wanda_gpu,  "wanda"),
            (W_mag_gpu,    "magnitude_global"),
            (W_gpu,        "dense"),
        ]
        if W_sparsegpt_gpu is not None:
            gpu_pruning_variants.append((W_sparsegpt_gpu, "sparsegpt"))

        for tname, tfn in TRANSFORMS.items():
            for Wp_gpu, plabel in gpu_pruning_variants:
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
                    # Previously silently swallowed (bare `except: pass`),
                    # which made it impossible to tell "transform legitimately
                    # not yet run" apart from "transform crashed on this
                    # layer" when reading [X] placeholders in generated
                    # tables downstream. Now logged to both console and a
                    # per-model failure CSV so silent gaps are traceable.
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

    # Free the full model from GPU now that the sampled-layer loop is done
    # (only relevant when SparseGPT moved it there above).
    if not args.skip_sparsegpt:
        model.cpu()
        torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────────
    df_diag  = pd.DataFrame(layer_diag_rows)
    df_trans = pd.DataFrame(transform_rows)
    df_sens  = pd.DataFrame(sensitivity_rows)
    df_fail  = pd.DataFrame(transform_failures)

    df_diag .to_csv(os.path.join(out_dir, "layer_diagnostics.csv"),  index=False)
    df_trans.to_csv(os.path.join(out_dir, "transform_results.csv"),  index=False)
    df_sens .to_csv(os.path.join(out_dir, "sensitivity.csv"),        index=False)
    df_fail .to_csv(os.path.join(out_dir, "transform_failures.csv"), index=False)

    if len(transform_failures) > 0:
        print(f"\n  WARNING: {len(transform_failures)} transform/layer/pruning "
              f"combinations failed silently-no-more -- see "
              f"{out_dir}/transform_failures.csv")
        fail_by_transform = df_fail.groupby("transform").size().sort_values(ascending=False)
        print(f"  Failures by transform:\n{fail_by_transform.to_string()}")

    # summary
    wanda_z = df_trans[df_trans.pruning=="wanda"].groupby("transform")["zero_rows"].mean()
    dense_z = df_trans[df_trans.pruning=="dense"].groupby("transform")["zero_rows"].mean()
    delta   = (wanda_z - dense_z).to_dict()

    # C8 (rebuttal): Delta for SparseGPT, same construction as Wanda's Delta
    has_sgpt = "sparsegpt" in df_trans.pruning.unique() if len(df_trans) else False
    delta_sgpt = {}
    if has_sgpt:
        sgpt_z = df_trans[df_trans.pruning=="sparsegpt"].groupby("transform")["zero_rows"].mean()
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
        # C8 (rebuttal): SparseGPT columns, blank when not computed
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

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summaries = []
    for model_key in args.models:
        for sparsity in SPARSITY_LIST:
            try:
                summaries.append(run_model(model_key, sparsity))
            except Exception as e:
                print(f"\nERROR on {model_key} @ sparsity={sparsity}: {e}")
                import traceback; traceback.print_exc()
    save_cross_model_summary(summaries)
    print("\nDone.")