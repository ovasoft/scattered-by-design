"""Unit tests for src/pruning_transforms. CPU only, a few seconds:
    python -m pytest tests/        (or: python tests/test_library.py)
"""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from pruning_transforms import config as C, metrics as M, pruning as P, transforms as T

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _layer(c_out=96, c_in=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    W  = torch.randn(c_out, c_in, generator=g)
    xn = torch.rand(c_in, generator=g) + 0.1
    return W, xn


def test_transform_count_and_order():
    assert len(T.TRANSFORMS) == 13
    assert list(T.BASELINE) == ["Identity"]
    assert list(T.ALL_TRANSFORMS)[0] == "Identity" and len(T.ALL_TRANSFORMS) == 14


def test_constants_unchanged():
    assert (C.ZERO_THRESH, C.ROW_THRESH, C.BLOCK_SIZE, C.NEAR_ZERO) == (1e-6, 0.99, 4, 0.01)
    assert C.SV_TOP_KS == [5, 10, 20, 50]
    assert C.RANK_THRESHOLDS == [1e-3, 1e-4, 1e-5, 1e-6]
    assert (C.N_CALIB, C.SEQ_LEN, C.N_BINS) == (128, 128, 10)


def test_all_transforms_run_on_tall_and_wide():
    # Thin factorisations (SVD_left/right/weighted, LU) return min(C_out, C_in)
    # rows or columns, so only the bound is checked.
    for c_out, c_in in [(96, 80), (80, 96)]:
        W, xn = _layer(c_out, c_in)
        Wp = P.wanda_prune_gpu(W, xn, 0.5)
        for name, fn in T.ALL_TRANSFORMS.items():
            out = fn(Wp, xn)
            assert out.ndim == 2 and out.shape[0] <= c_out and out.shape[1] <= c_in, (name, out.shape)
            assert np.isfinite(out).all(), name


def test_hadamard_stays_float32():
    W, xn = _layer(64, 100)                   # non power of two -> p = 128
    assert T.tg_hadamard_right(W, xn).dtype == np.float32


def test_col_sort_sorts_by_zero_fraction_not_activation():
    W = torch.ones(8, 4)
    W[:, 2] = 0.0                              # column 2 all zero
    W[:4, 0] = 0.0                             # column 0 half zero
    xn = torch.tensor([100.0, 1.0, 1.0, 50.0]) # activations must not matter
    out = T.tg_col_sort(W, xn)
    assert np.all(out[:, 0] == 0) and np.sum(out[:, 1] == 0) == 4


def test_rank_is_anchored_to_dense_sigma_max():
    W, xn = _layer()
    S = torch.linalg.svdvals(W)
    small = W * 1e-4                           # own sigma_max anchoring would keep full rank
    r = M.rank_at_thresholds(small, S[0].item())
    assert r["rank_1em3"] == 0 and r["rank_1em6"] == min(W.shape)


def test_negative_rank_drop_not_clamped():
    W, xn = _layer()
    smax = torch.linalg.svdvals(W)[0].item()
    d = M.rank_at_thresholds(W * 1e-4, smax)["rank_1em3"] - M.rank_at_thresholds(W, smax)["rank_1em3"]
    assert d < 0


def test_wanda_prunes_exact_fraction_per_row():
    W, xn = _layer()
    Wp = P.wanda_prune_gpu(W, xn, 0.5)
    assert torch.all((Wp == 0).sum(dim=1) == W.shape[1] // 2)


def test_zero_row_and_block_metrics():
    W = np.ones((8, 8), dtype=np.float32); W[:4] = 0.0
    assert M.zero_row_frac(W) == 0.5 and M.zero_col_frac(W) == 0.0
    assert M.block_sparsity_frac(W) == 0.5
    assert M.near_zero_row_frac(W) == 0.5


def test_sampled_blocks_match_published_csvs():
    # block_idx values in the published per-layer layer_diagnostics.csv files
    assert C.sampled_blocks(12) == [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]          # OPT-125M
    assert C.sampled_blocks(24) == [1, 3, 6, 8, 10, 13, 15, 18, 20, 22]      # OPT-1.3B
    assert C.sampled_blocks(32) == [1, 4, 8, 11, 14, 17, 20, 24, 27, 30]     # LLaMA-7B
    assert C.sampled_blocks(40) == [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]    # LLaMA-13B


def test_no_cholesky_reference_in_code():
    pat = re.compile("cholesky", re.I)
    for d in ("src", "experiments", "analysis"):
        for root, _, files in os.walk(os.path.join(REPO, d)):
            for f in files:
                if f.endswith(".py"):
                    src = open(os.path.join(root, f)).read()
                    # torch.linalg.cholesky inside the SparseGPT algorithm is not the transform
                    src = src.replace("torch.linalg.cholesky", "").replace("torch.cholesky_inverse", "")
                    assert not pat.search(src), f"Cholesky mention in {f}"


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
