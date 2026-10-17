#!/usr/bin/env bash
# End-to-end smoke test covering both pruning protocols and the synthetic
# ablation. Asserts known-good values and fails loudly on drift.
#
#   bash scripts/smoke_test.sh            # outputs go to a fresh temp dir
#   SMOKE_DIR=/some/dir bash scripts/smoke_test.sh
#
# Runtime on one RTX A6000: ~10 min (most of it the per-layer OPT-125M sweep
# and the official Wanda calibration pass). CPU-only works but is slower.
# Needs the HF Hub (or a populated HF cache) for facebook/opt-125m and C4.
#
# Tolerances are fixed by REPRODUCIBILITY_TASK.md section 7. Do not loosen
# them: a drift means an environment or logic change altered published
# numbers. Stop and investigate instead.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_DIR="${SMOKE_DIR:-$(mktemp -d -t sbd-smoke-XXXXXX)}"
mkdir -p "$SMOKE_DIR"
PY="${PYTHON:-python}"
# Cap BLAS threads: on a busy many-core node an uncapped numpy/scipy pool
# oversubscribes and runs ~10x slower. Override by exporting your own values.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
echo "smoke test outputs: $SMOKE_DIR"
cd "$REPO"

# ── 1. synthetic three-mechanism factorial (CPU) ─────────────────────────────
echo; echo "=== [1/3] synthetic_mechanism_ablation.py (flat/full cell) ==="
"$PY" experiments/synthetic_mechanism_ablation.py --spectrum flat --rank full \
    --out_dir "$SMOKE_DIR/synthetic"
"$PY" experiments/synthetic_mechanism_ablation.py --aggregate --out_dir "$SMOKE_DIR/synthetic"
"$PY" - "$SMOKE_DIR/synthetic" <<'EOF'
import sys, pandas as pd
g = pd.read_csv(f"{sys.argv[1]}/synthetic_mechanism_ablation_summary.csv")
cell = lambda m: g[(g["mask"] == m) & (g["spectrum"] == "flat") & (g["rank"] == "full")]["delta_svdleft_rows"]
checks = [("indep / flat / full", cell("indep"), -0.002),
          ("colshare_0.25 / flat / full", cell("colshare_0.25"), 0.25)]
bad = []
for name, v, ref in checks:
    assert len(v) == 1, f"missing synthetic cell {name}"
    v = float(v.iloc[0]); ok = abs(v - ref) <= 0.01
    print(f"  Delta(SVD-left) {name:<28} = {v:+.6f}   ref {ref:+.3f} +-0.01   {'OK' if ok else 'DRIFT'}")
    if not ok: bad.append(name)
if bad: sys.exit(f"SMOKE FAIL (synthetic): {bad}")
EOF

# ── 2. per-layer protocol, OPT-125M ──────────────────────────────────────────
echo; echo "=== [2/3] run_experiments.py --models opt-125m (per-layer protocol) ==="
"$PY" experiments/run_experiments.py --models opt-125m --results_dir "$SMOKE_DIR/perlayer"
"$PY" - "$SMOKE_DIR/perlayer/opt-125m" <<'EOF'
import sys, os, pandas as pd
d = sys.argv[1]
f = pd.read_csv(os.path.join(d, "transform_failures.csv"))
t = pd.read_csv(os.path.join(d, "transform_results.csv"))
n_t = t["transform"].nunique()
print(f"  transform_failures.csv rows: {len(f)}   transforms present: {n_t}/14")
if len(f): print(f.to_string(index=False)); sys.exit("SMOKE FAIL (per-layer): transform failures")
if n_t != 14: sys.exit("SMOKE FAIL (per-layer): expected 14 transforms (13 + Identity)")
EOF

# ── 3. whole-model protocol, OPT-125M, official Wanda ────────────────────────
echo; echo "=== [3/3] run_wholemodel_experiments.py (whole-model protocol) ==="
"$PY" experiments/run_wholemodel_experiments.py --wanda_repo third_party/wanda \
    --models opt-125m --methods wanda --sparsity 0.5 --seeds 0 \
    --results_dir "$SMOKE_DIR/wholemodel"
"$PY" - "$SMOKE_DIR/wholemodel/opt-125m/wanda/sp50/seed0" <<'EOF'
import sys, os, pandas as pd
d = sys.argv[1]
df = pd.read_csv(os.path.join(d, "delta_per_transform.csv"), index_col=0)
f  = pd.read_csv(os.path.join(d, "transform_failures.csv"))
TOL = 5e-4
REF = {"SVD_left": 0.0031467, "SVD_weighted": 0.0031467,
       "LU_left": 0.00017361, "QR_left": 0.00015191}   # all others exactly 0.0
bad = []
if len(df) != 14: bad.append(f"delta_per_transform.csv has {len(df)} rows, expected 14")
if len(f):        bad.append(f"{len(f)} transform failures")
best = df["zero_rows"].idxmax(); mx = df["zero_rows"].max()
print(f"  max Delta(zero_rows) = {mx:+.7f} at {best}   (ref +0.0031467 at SVD_left, tol {TOL})")
if best != "SVD_left" or abs(mx - 0.0031467) > TOL:
    bad.append(f"max Delta {mx} at {best}")
for t, v in df["zero_rows"].items():
    ref = REF.get(t, 0.0); ok = abs(v - ref) <= TOL
    print(f"    {t:<17} {v:+.8f}   ref {ref:+.8f}   {'OK' if ok else 'DRIFT'}")
    if not ok: bad.append(t)
if bad: sys.exit(f"SMOKE FAIL (whole-model): {bad}")
EOF

echo; echo "SMOKE TEST PASSED  (outputs in $SMOKE_DIR)"
