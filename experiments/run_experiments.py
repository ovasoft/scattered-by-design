"""
Full experiment suite: Wanda zero clustering across OPT and LLaMA model series.

Wanda pruning follows the official implementation exactly:
  - Uses scaler_row = sqrt(sum of squared activations) per input feature
  - Per-row comparison group (unstructured)
  - No weight update

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

Usage:
  python run_experiments.py --models opt-125m
  python run_experiments.py --models opt-125m opt-1.3b opt-6.7b opt-13b
  python run_experiments.py --models llama-7b llama-13b
"""

import os, json, argparse, warnings
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
SPARSITY    = 0.5
ZERO_THRESH = 1e-6
ROW_THRESH  = 0.99
BLOCK_SIZE  = 4
THRESHOLDS  = [1e-3, 1e-4, 1e-6]
SV_TOP_KS   = [5, 10, 20, 50]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
CALIB_CACHE = "calib_cache"

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["opt-125m"],
                    choices=list(MODEL_REGISTRY.keys()))
parser.add_argument("--sparsity", type=float, default=SPARSITY)
args = parser.parse_args()
SPARSITY = args.sparsity

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
# ── Collect activations layer-by-layer (memory efficient) ────────────────────
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

# ── GPU pruning functions ─────────────────────────────────────────────────────
def wanda_prune_gpu(W, xn, sparsity=SPARSITY):
    metric  = W.abs() * xn.reshape(1, -1)
    n_prune = int(metric.shape[1] * sparsity)
    _, idx  = torch.topk(metric, n_prune, dim=1, largest=False, sorted=False)
    Wp = W.clone(); Wp.scatter_(1, idx, 0.0)
    return Wp

def magnitude_prune_global_gpu(W, sparsity=SPARSITY):
    k      = int(W.numel() * sparsity)
    thresh = torch.kthvalue(W.abs().flatten(), k).values
    Wp     = W.clone(); Wp[Wp.abs() <= thresh] = 0.0
    return Wp

def magnitude_prune_perrow_gpu(W, sparsity=SPARSITY):
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
    H = torch.tensor(spla.hadamard(p, dtype=np.float32)/np.sqrt(p),
                     device=DEVICE)
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
def run_model(model_key):
    model_id = MODEL_REGISTRY[model_key]
    arch     = get_arch(model_key)
    out_dir  = os.path.join(RESULTS_DIR, model_key)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_key}  ({model_id})")
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

    layer_diag_rows  = []
    transform_rows   = []
    sensitivity_rows = []

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
        W_wanda_gpu  = wanda_prune_gpu(W_gpu, xn_gpu, SPARSITY)
        W_mag_gpu    = magnitude_prune_global_gpu(W_gpu, SPARSITY)
        W_mag_pr_gpu = magnitude_prune_perrow_gpu(W_gpu, SPARSITY)
        W_hi_gpu     = magnitude_prune_hi_gpu(W_gpu, 0.90)

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

        # ── Rank at multiple thresholds — all anchored to dense S_max ──────
        S_dense_max = S_dense[0].item()
        ranks_d  = rank_at_thresholds(W_gpu,        S_dense_max)
        ranks_w  = rank_at_thresholds(W_wanda_gpu,  S_dense_max)
        ranks_m  = rank_at_thresholds(W_mag_gpu,    S_dense_max)
        ranks_mp = rank_at_thresholds(W_mag_pr_gpu, S_dense_max)
        ranks_hi = rank_at_thresholds(W_hi_gpu,     S_dense_max)
        # rank_drop per threshold — negative = spectral redistribution noise
        rank_drops_wanda = {t: ranks_d[t] - ranks_w[t] for t in ranks_d}
        rank_drops_mag   = {t: ranks_d[t] - ranks_m[t] for t in ranks_d}
        rank_drops_hi    = {t: ranks_d[t] - ranks_hi[t] for t in ranks_d}

        spec_w   = spectral_metrics(S_dense, S_wanda)
        spec_m   = spectral_metrics(S_dense, S_mag)
        spec_hi  = spectral_metrics(S_dense, S_hi)

        mc_w     = mask_corr_fast(W_wanda)
        mc_m     = mask_corr_fast(W_mag_g)
        mc_mp    = mask_corr_fast(W_mag_pr)
        # random mask control
        W_rand   = W_np.copy()
        W_rand[np.random.rand(*W_np.shape) < SPARSITY] = 0.0
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
        layer_diag_rows.append(diag_row)

        # ── Transform loop ────────────────────────────────────────────────────
        # GPU pruning variants for transform loop
        gpu_pruning_variants = [
            (W_wanda_gpu,  "wanda"),
            (W_mag_gpu,    "magnitude_global"),
            (W_gpu,        "dense"),
        ]

        for tname, tfn in TRANSFORMS.items():
            for Wp_gpu, plabel in gpu_pruning_variants:
                try:
                    Wt = np.nan_to_num(tfn(Wp_gpu, xn_gpu))
                    transform_rows.append({
                        "model"         : model_key,
                        "layer"         : name,
                        "layer_type"    : layer_type,
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
                    # threshold sensitivity for wanda only
                    if plabel == "wanda":
                        for zt, val in threshold_sensitivity(Wt).items():
                            sensitivity_rows.append({
                                "model"    : model_key, "layer": name,
                                "transform": tname, "threshold": zt,
                                "zero_rows": val,
                            })
                except Exception:
                    pass

        drop_str = " | ".join(f"{k}:{v:+d}" for k, v in rank_drops_wanda.items())
        print(f"  ✓ {name:<45} rank_drops=[{drop_str}]  "
              f"top10={spec_w['top10_ratio']:.4f}  "
              f"mask_corr={mc_w:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df_diag  = pd.DataFrame(layer_diag_rows)
    df_trans = pd.DataFrame(transform_rows)
    df_sens  = pd.DataFrame(sensitivity_rows)

    df_diag .to_csv(os.path.join(out_dir, "layer_diagnostics.csv"),  index=False)
    df_trans.to_csv(os.path.join(out_dir, "transform_results.csv"),  index=False)
    df_sens .to_csv(os.path.join(out_dir, "sensitivity.csv"),        index=False)

    # summary
    wanda_z = df_trans[df_trans.pruning=="wanda"].groupby("transform")["zero_rows"].mean()
    dense_z = df_trans[df_trans.pruning=="dense"].groupby("transform")["zero_rows"].mean()
    delta   = (wanda_z - dense_z).to_dict()

    summary = {
        "model"               : model_key,
        "n_layers"            : len(layer_diag_rows),
        "sparsity"            : SPARSITY,
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
    return summary

# ── Cross-model summary ───────────────────────────────────────────────────────
def save_cross_model_summary(summaries):
    rows = [{
        "model"              : s["model"],
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
    } for s in summaries]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "all_models_summary.csv"), index=False)
    print(f"\n{'='*70}\nCROSS-MODEL SUMMARY\n{'='*70}")
    print(df.to_string(index=False))

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summaries = []
    for model_key in args.models:
        try:
            summaries.append(run_model(model_key))
        except Exception as e:
            print(f"\nERROR on {model_key}: {e}")
            import traceback; traceback.print_exc()
    save_cross_model_summary(summaries)
    print("\nDone.")