"""
The paper's post-hoc transformation suite: the single definition used by
every real-model experiment.

Convention: the paper reports 13 transformations, in ``TRANSFORMS``.
``BASELINE`` holds ``Identity``, the untransformed matrix. It is a reference
point, not one of the 13. ``ALL_TRANSFORMS`` is baseline plus suite (14
entries) in the order the experiment scripts iterate, which is the order of
rows in every ``transform_results.csv``.

Every transform has the signature ``t(Wp, xn) -> np.ndarray``:
  Wp : torch float32 tensor (C_out, C_in), dense or pruned weight
  xn : torch float32 tensor (C_in,), Wanda activation scaler sqrt(mean ||X_j||^2)
       from the DENSE model. Used only by the activation-aware transforms
       (PCA_right, SVD_weighted, Diagonal_scaling).
The result is a numpy array on the CPU. Callers apply ``np.nan_to_num``.
"""
import numpy as np
import torch
from scipy import linalg as spla
from scipy.fft import dct
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from sklearn.decomposition import PCA as skPCA

from .config import ZERO_THRESH


def tg_identity(Wp, xn):
    return Wp.cpu().numpy()


def tg_pca_right(Wp, xn):
    """PCA rotation of the input space, W -> W Q. Q is the PCA basis of a
    synthetic activation proxy (C_in + 10 Gaussian samples with column std
    xn, fixed seed 42). Using C_in + 10 samples makes PCA return a full-rank
    (C_in, C_in) rotation."""
    xn_np = xn.cpu().numpy()
    C_in  = len(xn_np)
    rng   = np.random.default_rng(42)
    X     = rng.standard_normal((C_in + 10, C_in)).astype(np.float32) * xn_np
    pca   = skPCA(n_components=C_in)
    pca.fit(X)
    Q     = pca.components_.T                     # (C_in, C_in)
    return (Wp @ torch.tensor(Q, device=Wp.device)).cpu().numpy()


def tg_svd_right(Wp, xn):
    """W -> W V (right singular basis of W itself)."""
    _, _, Vt = torch.linalg.svd(Wp, full_matrices=False)
    return (Wp @ Vt.T).cpu().numpy()


def tg_svd_left(Wp, xn):
    """W -> U^T W = S V^T (left singular basis). Rows of S V^T are zero
    exactly where singular values vanish, so this is the transform that
    exposes rank loss as zero rows."""
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
    return (torch.diag(S) @ Vt).cpu().numpy()


def tg_svd_twosided(Wp, xn):
    """W -> U^T W V = diag(S), embedded in a (C_out, C_in) zero matrix."""
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
    W2 = torch.zeros_like(Wp)
    k  = S.shape[0]
    W2[:k, :k] = torch.diag(S)
    return W2.cpu().numpy()


def tg_qr_left(Wp, xn):
    """W -> R from the complete QR decomposition W = Q R."""
    Q, R = torch.linalg.qr(Wp, mode='complete')
    return R.cpu().numpy()


def tg_lu_left(Wp, xn):
    """W -> U from the LU decomposition P L U = W (scipy, partial pivoting)."""
    _, _, U = spla.lu(Wp.cpu().numpy())
    return U


def tg_hadamard_right(Wp, xn):
    """W -> W H[:n, :n], with H the normalised Sylvester Hadamard matrix of
    the next power of two p >= n = C_in.

    Dtype: ``spla.hadamard(p, dtype=np.float32) / np.sqrt(p)`` is promoted to
    float64 by NumPy scalar promotion (np.sqrt of a Python int is a
    np.float64). That breaks the float32 torch matmul. The ``.astype(np.float32)``
    after the division is required."""
    n = Wp.shape[1]
    p = int(2**np.ceil(np.log2(n)))
    H_np = (spla.hadamard(p, dtype=np.float32) / np.sqrt(p)).astype(np.float32)
    H = torch.tensor(H_np, device=Wp.device)
    return (Wp @ H[:n, :n]).cpu().numpy()


def tg_dct_right(Wp, xn):
    """W -> W D^T, D the orthonormal DCT-II matrix of size C_in."""
    n = Wp.shape[1]
    Q = torch.tensor(dct(np.eye(n, dtype=np.float32), axis=0, norm='ortho').T,
                     device=Wp.device)
    return (Wp @ Q).cpu().numpy()


def tg_diagonal_scaling(Wp, xn):
    """W -> W diag(xn) (the Wanda importance rescaling of input columns)."""
    return (Wp * xn.reshape(1, -1)).cpu().numpy()


def tg_col_sort(Wp, xn):
    """Column permutation: reorder the columns of W by their fraction of
    exactly-zero entries (|w| < ZERO_THRESH), most-zero first, using a
    descending torch.argsort. It does NOT sort by activation magnitude: xn is
    ignored. (The manuscript text says "column sorting by activation
    magnitude"; this zero-fraction sort is what produced every reported
    number.)"""
    zero_frac = (Wp.abs() < ZERO_THRESH).float().mean(dim=0)
    idx = torch.argsort(zero_frac, descending=True)
    return Wp[:, idx].cpu().numpy()


def tg_row_sort(Wp, xn):
    """Row permutation: reorder rows by their fraction of exactly-zero
    entries (|w| < ZERO_THRESH), most-zero first."""
    zero_frac = (Wp.abs() < ZERO_THRESH).float().mean(dim=1)
    idx = torch.argsort(zero_frac, descending=True)
    return Wp[idx].cpu().numpy()


def tg_svd_weighted(Wp, xn):
    """Activation-weighted SVD-left: S V^T of W diag(xn)."""
    U, S, Vt = torch.linalg.svd(Wp * xn.reshape(1, -1), full_matrices=False)
    return (torch.diag(S) @ Vt).cpu().numpy()


def tg_cuthill_mckee(Wp, xn):
    """Reverse Cuthill-McKee row reordering of the non-zero pattern. Defined
    only for square matrices: non-square matrices are returned unchanged."""
    W_np = Wp.cpu().numpy()
    adj  = csr_matrix((np.abs(W_np) > ZERO_THRESH).astype(np.float32))
    if adj.shape[0] == adj.shape[1]:
        return W_np[reverse_cuthill_mckee(adj)]
    return W_np


BASELINE = {"Identity": tg_identity}

# The 13 transformations reported in the paper.
TRANSFORMS = {
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

ALL_TRANSFORMS = {**BASELINE, **TRANSFORMS}

assert len(TRANSFORMS) == 13 and len(ALL_TRANSFORMS) == 14
