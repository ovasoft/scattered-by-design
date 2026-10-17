"""
Thresholds and constants shared by every experiment.

These values produced every number in the paper. Do not change them: changing
any one of them changes published results. Every script imports from here so
the scripts cannot silently disagree.
"""
from pathlib import Path

import numpy as np

# ── Zero / structure thresholds ──────────────────────────────────────────────
ZERO_THRESH = 1e-6      # |w| below this counts as zero
ROW_THRESH  = 0.99      # fraction of a row that must be zero to count as a zero row
                        # (also used for zero columns)
BLOCK_SIZE  = 4         # for 4x4 block sparsity
BLOCK_THRESH = 0.9      # a 4x4 block counts as sparse if >= 90% of it is zero
NEAR_ZERO   = 0.01      # slice norm < 1% of mean norm counts as deletable (SliceGPT-style)

# Synthetic ablation only: synthetic matrices have arbitrary scale, so it uses
# the scale-equivalent RELATIVE zero threshold  SYNTH_REL_ZERO * max|T(W)|
# instead of the absolute ZERO_THRESH used on trained weights.
SYNTH_REL_ZERO = 1e-6

# ── Spectral diagnostics ─────────────────────────────────────────────────────
SV_TOP_KS   = [5, 10, 20, 50]
# Rank thresholds are ALWAYS anchored to sigma_max(W_dense): a singular value
# of the dense OR pruned matrix counts towards the rank iff it exceeds
# t * sigma_max(W_dense). Anchoring to each matrix's own sigma_max produces
# spurious rank differences between dense and pruned (stated in the paper).
# Negative rank drops (pruned has MORE singular values above the threshold
# than dense) are real spectral redistribution, not a bug: never clamp them.
RANK_THRESHOLDS = [1e-3, 1e-4, 1e-5, 1e-6]
RANK_TAGS = {1e-3: "1em3", 1e-4: "1em4", 1e-5: "1em5", 1e-6: "1em6"}  # CSV column suffixes

# Zero thresholds swept in the per-layer threshold-sensitivity table.
SENSITIVITY_THRESHOLDS = [1e-3, 1e-4, 1e-6]

# ── Calibration ──────────────────────────────────────────────────────────────
N_CALIB     = 128
SEQ_LEN     = 128       # per-layer pipeline only; whole-model uses model.seqlen (2048)
N_BINS      = 10        # depth-stratified sampling: 10 bins, central block of each
SPARSITY    = 0.5       # default target sparsity

# Layers outside this size window are skipped by every real-model analysis.
MIN_DIM, MAX_C_OUT = 64, 16384

# ── Models ───────────────────────────────────────────────────────────────────
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

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WANDA_REPO = REPO_ROOT / "third_party" / "wanda"
RESULTS_ROOT = REPO_ROOT / "results"
CALIB_CACHE = REPO_ROOT / "calib_cache"


def get_arch(model_key):
    return "llama" if "llama" in model_key else "opt"


def is_eligible(name, arch):
    """All linear layers except embeddings and lm_head."""
    return (any(p in name for p in LAYER_PATTERNS[arch])
            and "lm_head" not in name)


def sampled_blocks(n_blocks, n_bins=N_BINS):
    """Depth-stratified sampling: split depth into n_bins equal bins and take
    the central block of each (deduplicated for shallow models). Every linear
    layer type within a sampled block is analysed."""
    edges = np.linspace(0, n_blocks, n_bins + 1)
    return sorted(set(min(int((edges[b] + edges[b + 1]) / 2), n_blocks - 1)
                      for b in range(n_bins)))
