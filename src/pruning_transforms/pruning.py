"""
Pruning functions and calibration statistics.

Two protocols use this module (see README, "Two pruning protocols"):

* Per-layer protocol (experiments/run_experiments*.py): each sampled layer is
  pruned in isolation by the pure functions below, using Wanda scalers
  collected from ONE forward pass of the DENSE model (deliberate: do not
  change it to sequential calibration).
* Whole-model protocol (run_wholemodel_experiments.py, run_crosslayer_q.py,
  run_oproj_mechanism.py): pruning is delegated to the official Wanda /
  SparseGPT code in third_party/wanda (``import_official_pruners``). The only
  pruner defined here for that protocol is per-row magnitude, which has no
  official implementation.
"""
import math
import os
import sys

import torch
import torch.nn as nn

from .config import DEFAULT_WANDA_REPO, is_eligible


# ── Wanda activation scaler (mirrors official WrappedGPT) ────────────────────
class WrappedLayer:
    """Mirrors the official Wanda ``WrappedGPT`` exactly. Keeps an online
    running mean
        scaler_row_j = mean over calibration tokens of X_j^2
    so ``get_scaler()`` = sqrt(scaler_row) and the Wanda metric is
    |W_ij| * get_scaler()_j."""
    def __init__(self, layer):
        self.layer      = layer
        self.dev        = layer.weight.device
        self.rows       = layer.weight.data.shape[0]
        self.columns    = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.nsamples   = 0

    def add_batch(self, inp, out=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()                  # (C_in, N*L)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples   += tmp
        inp = inp.float()
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def get_scaler(self):
        return torch.sqrt(self.scaler_row)


def collect_scalers(model, batches, keep):
    """One forward pass over ``batches`` with a WrappedLayer hook on every
    nn.Linear whose name satisfies ``keep(name)``. Returns
    {layer_name: scaler (C_in,) CPU tensor}."""
    wrapped, handles = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not keep(name):
            continue
        wl = WrappedLayer(module)
        wrapped[name] = wl
        def make_hook(w):
            def hook(m, inp, out):
                w.add_batch(inp[0].detach())
            return hook
        handles.append(module.register_forward_hook(make_hook(wl)))
    dev = next(model.parameters()).device
    with torch.no_grad():
        for b in batches:
            model(b.to(dev))
    for h in handles:
        h.remove()
    return {n: w.get_scaler().cpu() for n, w in wrapped.items()}


# ── Per-layer pruning (pure functions: return a new tensor) ──────────────────
def wanda_prune_gpu(W, xn, sparsity):
    """Wanda: per row, zero the ``sparsity`` fraction with smallest |W|*xn."""
    metric  = W.abs() * xn.reshape(1, -1)
    n_prune = int(metric.shape[1] * sparsity)
    _, idx  = torch.topk(metric, n_prune, dim=1, largest=False, sorted=False)
    Wp = W.clone(); Wp.scatter_(1, idx, 0.0)
    return Wp


def magnitude_prune_global_gpu(W, sparsity):
    """Global (whole-matrix) magnitude pruning: zero every |w| <= the
    k-th smallest magnitude, k = sparsity * numel."""
    k      = int(W.numel() * sparsity)
    thresh = torch.kthvalue(W.abs().flatten(), k).values
    Wp     = W.clone(); Wp[Wp.abs() <= thresh] = 0.0
    return Wp


def magnitude_prune_perrow_gpu(W, sparsity):
    """Per-row (per-output) magnitude pruning: Wanda's comparison group
    without activation statistics."""
    n_prune = int(W.shape[1] * sparsity)
    _, idx  = torch.topk(W.abs(), n_prune, dim=1, largest=False, sorted=False)
    Wp      = W.clone(); Wp.scatter_(1, idx, 0.0)
    return Wp


def magnitude_prune_hi_gpu(W, sparsity=0.90):
    """High-sparsity global magnitude control (validates the rank-drop
    diagnostic in a regime where rank loss does occur)."""
    return magnitude_prune_global_gpu(W, sparsity)


# ── SparseGPT, per-layer (official IST-DASLab/sparsegpt, trimmed) ────────────
class SparseGPTWrapped:
    """Adapted from the official SparseGPT class. The optional quantizer hooks
    and Conv2d/Conv1D branches are removed, since only nn.Linear layers are
    pruned. The core algorithm (Hessian accumulation + sequential per-block
    OBS update) is unchanged from the reference implementation.

    Unlike WrappedLayer (Wanda), this accumulates a full (C_in, C_in) Hessian,
    and ``fasterprune`` performs a sequential weight UPDATE (not just a mask)
    that propagates pruning error to not-yet-pruned weights."""
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
        """Returns the pruned weight (C_out, C_in) as a tensor on self.dev.
        Does NOT mutate self.layer.weight. The official SparseGPT writes in
        place; this pipeline treats pruning functions as pure."""
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
    """Minimal stand-in exposing .weight.data, .weight.device and
    .weight.shape, so SparseGPTWrapped can wrap a raw weight tensor without
    touching the live model's nn.Linear module."""
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


def sparsegpt_prune_layer(model, module, calib_tokens, sparsity, device,
                          blocksize=128, percdamp=0.01):
    """SparseGPT-prune one layer in isolation (per-layer protocol). The
    Hessian comes from a fresh forward pass of the full (dense) model, with a
    hook on ``module`` only, in mini-batches of 4. ``model`` must already be
    on ``device``. Returns the pruned weight on ``device``. Only one layer is
    hooked at a time because H is O(C_in^2) per layer."""
    shim = _LinearShim(module.weight.data.to(device))
    sgpt = SparseGPTWrapped(shim)

    def hook(m, inp, out):
        sgpt.add_batch(inp[0].detach())

    handle = module.register_forward_hook(hook)
    calib = calib_tokens.to(device)
    try:
        with torch.no_grad():
            for i in range(0, len(calib), 4):
                model(calib[i:i+4])
    finally:
        handle.remove()

    W_pruned = sgpt.fasterprune(sparsity, blocksize=blocksize, percdamp=percdamp)
    sgpt.free()
    return W_pruned.to(device)


# ── Whole-model per-row magnitude (no official implementation exists) ────────
def prune_magnitude_perrow(model, arch, sparsity):
    """In place, per row of every eligible linear layer, zero the smallest
    ``sparsity`` fraction of weights by |W|. Same per-output comparison group
    as Wanda/SparseGPT, but no activation statistics and no weight update.
    Deterministic (no calibration)."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and is_eligible(name, arch):
            W = m.weight.data
            n_prune = int(W.shape[1] * sparsity)
            if n_prune <= 0:
                continue
            _, idx = torch.topk(W.abs(), n_prune, dim=1,
                                largest=False, sorted=False)
            W.scatter_(1, idx, 0.0)


# ── Official locuslab/wanda pruners (vendored in third_party/wanda) ──────────
def import_official_pruners(arch, repo_path=DEFAULT_WANDA_REPO):
    """Returns a dict with prune_wanda, prune_sparsegpt, prune_magnitude,
    check_sparsity, eval_ppl and get_loaders from the official Wanda tree.
    The official repo splits LLaMA (lib/prune.py) and OPT (lib/prune_opt.py)
    support. Pruners and eval are imported separately, so a missing
    lib/eval_opt.py can never make OPT fall back to the LLaMA pruners."""
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(os.path.join(repo_path, "lib")):
        raise FileNotFoundError(f"no Wanda checkout at {repo_path}")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from lib.data import get_loaders
    if arch == "llama":
        from lib.prune import (prune_wanda, prune_sparsegpt,
                               prune_magnitude, check_sparsity)
        from lib.eval import eval_ppl
    else:
        from lib.prune_opt import (prune_wanda, prune_sparsegpt,
                                   prune_magnitude, check_sparsity)
        try:
            from lib.eval_opt import eval_ppl
        except ImportError:
            from lib.eval import eval_ppl
    return dict(prune_wanda=prune_wanda, prune_sparsegpt=prune_sparsegpt,
                prune_magnitude=prune_magnitude, check_sparsity=check_sparsity,
                eval_ppl=eval_ppl, get_loaders=get_loaders)
