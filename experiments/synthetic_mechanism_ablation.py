"""
E5 (rebuttal, R1-Q5 / R4): controlled synthetic ablation of the three mechanisms.

Design. Synthetic weight matrices W = U diag(sigma) V^T (n x n, random
orthogonal U, V), with three independently controlled knobs:

  SPECTRUM (mechanism 2, spectral dispersion):
    flat          sigma_i = 1                      (dispersed  - satisfied)
    concentrated  sigma_i = exp(-8 (i-1)/n)        (concentrated - violated)

  RANK (mechanism 1, rank preservation):
    full          all n singular values as above    (preserved - satisfied)
    half          sigma_i = 0 for i > n/2           (deficient - violated)

  MASK (mechanism 3, mask independence), all at 50% overall sparsity:
    indep         per-row independent random 50% of columns (rho ~ 0, Wanda-like)
    group_G       rows split into G groups; each group shares one random
                  50%-column mask. G=32/8: moderate-to-high rho(M) with the
                  column-support union covering all columns => correlation
                  WITHOUT rank drop. G=2: high rho AND support-union shrinkage
                  => rank drop co-occurs.
    colshare_c    a fraction c of columns is zeroed in EVERY row (c in
                  {0.10, 0.25, 0.40}), per-row random fill to 50%. Jointly
                  violating: correlation + guaranteed rank drop.

Delta is the paper's marginal-contribution metric:
  Delta = metric(T(W_masked)) - metric(T(W_dense))

Zero threshold: the paper uses an absolute 1e-6 on trained weights; synthetic
scales differ, so we use the scale-equivalent relative threshold
1e-6 * max|T(W)|.

Pure numpy; runs on CPU in well under a minute.
"""

import numpy as np
import pandas as pd

N        = 512
SPARSITY = 0.5
SEEDS    = [0, 1, 2, 3, 4]
ROW_THRESH = 0.99
BLOCK    = 4
RANK_T   = 1e-3     # rank threshold, anchored to sigma_max(dense)

# ── matrix construction ──────────────────────────────────────────────────────
def rand_orth(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q

def make_dense(spectrum, rank, rng, n=N):
    i = np.arange(n)
    sigma = np.ones(n) if spectrum == "flat" else np.exp(-8.0 * i / n)
    if rank == "half":
        sigma[n // 2:] = 0.0
    return rand_orth(n, rng) @ np.diag(sigma) @ rand_orth(n, rng).T

# ── mask construction (all 50% zeros) ────────────────────────────────────────
def make_mask(kind, rng, n=N, sparsity=SPARSITY):
    k = int(n * sparsity)
    M = np.ones((n, n), dtype=bool)          # True = retained
    if kind == "indep":
        for r in range(n):
            M[r, rng.choice(n, k, replace=False)] = False
    elif kind.startswith("group_"):
        n_groups = int(kind.split("_")[1])
        for g in range(n_groups):
            cols = rng.choice(n, k, replace=False)
            rows = np.where(np.arange(n) % n_groups == g)[0]
            for r in rows:
                M[r, cols] = False
    elif kind.startswith("colshare_"):
        c = float(kind.split("_")[1])
        shared = rng.choice(n, int(n * c), replace=False)
        M[:, shared] = False
        per_row_extra = k - len(shared)
        rest = np.setdiff1d(np.arange(n), shared)
        for r in range(n):
            M[r, rng.choice(rest, per_row_extra, replace=False)] = False
    else:
        raise ValueError(kind)
    return M

# ── metrics (paper definitions, relative zero threshold) ─────────────────────
def zt_of(W):
    return 1e-6 * (np.abs(W).max() + 1e-30)

def zero_rows(W):
    zt = zt_of(W)
    return float(((np.abs(W) < zt).mean(axis=1) >= ROW_THRESH).mean())

def zero_cols(W):
    zt = zt_of(W)
    return float(((np.abs(W) < zt).mean(axis=0) >= ROW_THRESH).mean())

def block4x4(W, bs=BLOCK, thresh=0.9):
    zt = zt_of(W)
    nr, nc = W.shape[0] // bs, W.shape[1] // bs
    blocks = np.abs(W[:nr*bs, :nc*bs]).reshape(nr, bs, nc, bs)
    return float(((blocks < zt).mean(axis=(1, 3)) >= thresh).mean())

def mask_corr(M, rng, max_rows=256):
    m = (~M).astype(np.float64)
    idx = rng.choice(m.shape[0], min(max_rows, m.shape[0]), replace=False)
    m = m[idx]
    m -= m.mean(axis=1, keepdims=True)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    corr = m @ m.T
    return float(np.mean(np.abs(corr[np.triu_indices(len(corr), k=1)])))

def rank_at(W, smax_dense):
    S = np.linalg.svd(W, compute_uv=False)
    return int((S > RANK_T * smax_dense).sum())

def top10_ratio(W_pruned, W_dense):
    Sp = np.linalg.svd(W_pruned, compute_uv=False)
    Sd = np.linalg.svd(W_dense, compute_uv=False)
    return float(Sp[:10].sum() / (Sd[:10].sum() + 1e-30))

# ── transforms (subset of the paper's suite; numpy) ──────────────────────────
def t_identity(W): return W
def t_svd_left(W):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return np.diag(S) @ Vt
def t_qr_left(W):
    _, R = np.linalg.qr(W, mode='complete')
    return R
def t_row_sort(W):
    zt = zt_of(W)
    return W[np.argsort(-(np.abs(W) < zt).mean(axis=1))]
def t_col_sort(W):
    zt = zt_of(W)
    return W[:, np.argsort(-(np.abs(W) < zt).mean(axis=0))]

TRANSFORMS = {"Identity": t_identity, "SVD_left": t_svd_left,
              "QR_left": t_qr_left, "Row_sort": t_row_sort,
              "Col_sort": t_col_sort}
METRICS = {"zero_rows": zero_rows, "zero_cols": zero_cols, "block4x4": block4x4}

# ── run factorial ────────────────────────────────────────────────────────────
MASK_KINDS = ["indep", "group_32", "group_8", "group_2",
              "colshare_0.10", "colshare_0.25", "colshare_0.40"]

def run_cell(spectrum, rank):
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 * seed + 7)
        W    = make_dense(spectrum, rank, rng)
        smax = np.linalg.svd(W, compute_uv=False)[0]
        rank_dense = rank_at(W, smax)
        Sd10 = np.linalg.svd(W, compute_uv=False)[:10].sum()
        dense_metrics = {}
        for tname, tfn in TRANSFORMS.items():
            Td = tfn(W)
            for mname, mfn in METRICS.items():
                dense_metrics[(tname, mname)] = mfn(Td)

        for mkind in MASK_KINDS:
            M  = make_mask(mkind, rng)
            Wp = W * M
            Sp = np.linalg.svd(Wp, compute_uv=False)
            diag = {
                "spectrum": spectrum, "rank": rank, "mask": mkind, "seed": seed,
                "rho_M": mask_corr(M, rng),
                "rank_dense": rank_dense,
                "rank_pruned": int((Sp > RANK_T * smax).sum()),
                "R10": float(Sp[:10].sum() / (Sd10 + 1e-30)),
            }
            diag["rank_drop"] = diag["rank_dense"] - diag["rank_pruned"]
            for tname, tfn in TRANSFORMS.items():
                Tp = tfn(Wp)
                for mname, mfn in METRICS.items():
                    diag[f"delta_{tname}_{mname}"] = (mfn(Tp)
                                                      - dense_metrics[(tname, mname)])
            rows.append(diag)
    return rows

def main():
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", choices=["flat", "concentrated"], default=None)
    ap.add_argument("--rank", choices=["full", "half"], default=None)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()

    if not a.aggregate:
        spectra = [a.spectrum] if a.spectrum else ["flat", "concentrated"]
        ranks   = [a.rank] if a.rank else ["full", "half"]
        rows = []
        for spectrum in spectra:
            for rank in ranks:
                print(f"cell {spectrum}/{rank} ...", flush=True)
                rows += run_cell(spectrum, rank)
        part = f"e5_part_{'-'.join(spectra)}_{'-'.join(ranks)}.csv"
        pd.DataFrame(rows).to_csv(part, index=False)
        print(f"wrote {part}")
        if a.spectrum or a.rank:
            return

    parts = sorted(glob.glob("e5_part_*.csv"))
    if not parts:
        print("no part files to aggregate"); return
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(
        subset=["spectrum", "rank", "mask", "seed"])
    df.to_csv("synthetic_mechanism_ablation_raw.csv", index=False)

    delta_row_cols = [c for c in df.columns if c.startswith("delta_")
                      and c.endswith("zero_rows")]
    df["best_delta_rows"] = df[delta_row_cols].max(axis=1)

    g = (df.groupby(["mask", "spectrum", "rank"])
           .agg(rho_M=("rho_M", "mean"),
                rank_drop=("rank_drop", "mean"),
                R10=("R10", "mean"),
                delta_svdleft_rows=("delta_SVD_left_zero_rows", "mean"),
                delta_svdleft_rows_std=("delta_SVD_left_zero_rows", "std"),
                best_delta_rows=("best_delta_rows", "mean"))
           .reset_index()
           .sort_values(["mask", "spectrum", "rank"]))
    g.to_csv("synthetic_mechanism_ablation_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(g.round(4).to_string(index=False))

    base = df[(df["mask"] == "indep") & (df["spectrum"] == "flat")
              & (df["rank"] == "full")]["delta_SVD_left_zero_rows"]
    print(f"\nWanda-like cell (indep/flat/full): "
          f"Delta_SVDleft_rows = {base.mean():.6f} +/- {base.std():.6f}")

if __name__ == "__main__":
    main()