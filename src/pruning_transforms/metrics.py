"""
Structured-sparsity and spectral metrics, shared by every experiment.

Delta, the paper's central metric, is always built from these:
    Delta(T, m) = m(T(W_pruned)) - m(T(W_dense))
"""
import numpy as np
import torch

from .config import (ZERO_THRESH, ROW_THRESH, BLOCK_SIZE, BLOCK_THRESH,
                     NEAR_ZERO, SV_TOP_KS, RANK_THRESHOLDS, RANK_TAGS,
                     SENSITIVITY_THRESHOLDS)


# ── Structured sparsity (numpy, on transformed matrices) ─────────────────────
def zero_row_frac(W, zt=ZERO_THRESH, rt=ROW_THRESH):
    """Fraction of rows with at least ``rt`` of their entries below ``zt``."""
    return float(((np.abs(W) < zt).mean(axis=1) >= rt).mean())


def zero_col_frac(W, zt=ZERO_THRESH, rt=ROW_THRESH):
    """Fraction of columns with at least ``rt`` of their entries below ``zt``."""
    return float(((np.abs(W) < zt).mean(axis=0) >= rt).mean())


def near_zero_row_frac(W, thresh=NEAR_ZERO):
    """Fraction of rows whose L2 norm is below ``thresh`` x the mean row norm
    (SliceGPT-style deletable slices)."""
    norms = np.linalg.norm(W, axis=1)
    return float((norms < thresh * (norms.mean() + 1e-12)).mean())


def near_zero_col_frac(W, thresh=NEAR_ZERO):
    """Column analogue of ``near_zero_row_frac``."""
    norms = np.linalg.norm(W, axis=0)
    return float((norms < thresh * (norms.mean() + 1e-12)).mean())


def block_sparsity_frac(W, bs=BLOCK_SIZE, thresh=BLOCK_THRESH, zt=ZERO_THRESH):
    """Fraction of non-overlapping bs x bs blocks that are at least ``thresh``
    zero (trailing partial blocks are ignored)."""
    C_out, C_in = W.shape
    nr, nc = C_out // bs, C_in // bs
    if nr == 0 or nc == 0:
        return 0.0
    blocks = np.abs(W[:nr*bs, :nc*bs]).reshape(nr, bs, nc, bs)
    zero_frac = (blocks < zt).mean(axis=(1, 3))   # (nr, nc)
    return float((zero_frac >= thresh).mean())


def threshold_sensitivity(W, thresholds=SENSITIVITY_THRESHOLDS):
    """zero_row_frac of W at each absolute zero threshold."""
    return {zt: zero_row_frac(W, zt=zt) for zt in thresholds}


def qr_theory_zero_rows(C_out, C_in):
    """Zero rows that QR's R factor has by construction (tall matrices)."""
    return float(max(0, C_out - C_in) / C_out)


# ── Rank and spectrum (torch) ────────────────────────────────────────────────
def rank_at_thresholds(W, S_dense_max, thresholds=RANK_THRESHOLDS):
    """Rank of W at each threshold t: #{sigma_i(W) > t * S_dense_max}.

    ``S_dense_max`` MUST be sigma_max of the DENSE matrix, also when W is
    the pruned matrix. Keys: rank_1em3, rank_1em4, rank_1em5, rank_1em6."""
    S = torch.linalg.svdvals(W)
    return {f"rank_{RANK_TAGS[t]}": int((S > S_dense_max * t).sum().item())
            for t in thresholds}


def spectral_metrics(S_dense, S_pruned, top_ks=SV_TOP_KS):
    """Effective rank (exp of singular-value entropy) of dense and pruned, and
    the top-k singular-value energy ratio sum_k S_pruned / sum_k S_dense.
    Takes pre-computed singular values (torch tensors)."""
    def eff_rank(S):
        s = S / (S.sum() + 1e-12) + 1e-15
        return float(torch.exp(-(s * torch.log(s)).sum()).item())
    out = {"eff_rank_dense": eff_rank(S_dense), "eff_rank_pruned": eff_rank(S_pruned)}
    for k in top_ks:
        kk = min(k, S_dense.shape[0], S_pruned.shape[0])
        out[f"top{k}_ratio"] = float(
            S_pruned[:kk].sum().item() / (S_dense[:kk].sum().item() + 1e-12))
    return out


# ── Inter-row mask correlation ───────────────────────────────────────────────
def mask_corr(W_pruned, max_rows=256, rng=None, zt=ZERO_THRESH):
    """Mean absolute Pearson correlation between the zero masks of pairs of
    rows (rho(M) in the paper). Computed as normalised dot products of
    centred masks on at most ``max_rows`` randomly chosen rows.

    rng=None draws the row subsample from NumPy's global RNG (np.random),
    which is what the per-layer and whole-model pipelines did. Pass a
    ``np.random.Generator`` for a reproducible draw (out_proj mechanism
    analysis)."""
    mask = (np.abs(W_pruned) < zt).astype(np.float32)
    if mask.shape[0] > max_rows:
        chooser = np.random if rng is None else rng
        idx = chooser.choice(mask.shape[0], max_rows, replace=False)
        mask = mask[idx]
    mask -= mask.mean(axis=1, keepdims=True)
    mask /= (np.linalg.norm(mask, axis=1, keepdims=True) + 1e-12)
    corr = mask @ mask.T
    off  = corr[np.triu_indices(len(corr), k=1)]
    return float(np.mean(np.abs(off)))
