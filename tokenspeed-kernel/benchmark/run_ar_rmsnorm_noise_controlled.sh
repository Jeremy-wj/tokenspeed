#!/usr/bin/env bash
# Noise-controlled driver for the fused AR+RMSNorm backend benchmark.
#
# Mirrors the triton-shmem repo's noise-reverification protocol
# (results/ar_rmsnorm_opt_sweep/07_noise_reverification.md), adapted to this box:
#   * World-size isolation: each ws runs in its OWN `python -m ...` process with a
#     cooldown between, so a long sweep can't heat-soak later world sizes.
#   * High warmup/repeat (30/150 by default) -- large tensors settle to <2% noise.
#   * Two full passes -> a run-to-run VARIANCE FLOOR so conclusions are drawn only
#     from configs proven stable (small/latency-bound configs are inherently noisy).
#   * Best-effort GPU clock pin. NOTE: on the unprivileged migration container the
#     sysfs perf-control node is read-only, so the pin is a NO-OP here -- hence the
#     reliance on the 2-pass variance floor above. Run in a privileged container to
#     actually lock clocks.
#
# Usage (from the tokenspeed-kernel repo root, inside the ROCm container):
#   bash benchmark/run_ar_rmsnorm_noise_controlled.sh
# Env overrides: WORLD_SIZES, BENCH_BACKENDS, BENCH_M_VALUES, BENCH_N_VALUES,
#   BENCH_N_WARMUP, BENCH_N_REPEAT, COOLDOWN, OUTDIR.
set -uo pipefail
cd "$(dirname "$0")/.."

WORLD_SIZES="${WORLD_SIZES:-2 4 8}"
export BENCH_BACKENDS="${BENCH_BACKENDS:-rccl_unfused,triton_shmem}"
export BENCH_N_WARMUP="${BENCH_N_WARMUP:-30}"
export BENCH_N_REPEAT="${BENCH_N_REPEAT:-150}"
COOLDOWN="${COOLDOWN:-20}"
OUTDIR="${OUTDIR:-results/ar_rmsnorm_noise_controlled}"
mkdir -p "$OUTDIR"

before=$(rocm-smi --showperflevel 2>/dev/null | grep -m1 -o 'Performance Level: .*' || true)
rocm-smi --setperfdeterminism 2100 >/dev/null 2>&1 || true
after=$(rocm-smi --showperflevel 2>/dev/null | grep -m1 -o 'Performance Level: .*' || true)
if [ "$before" = "$after" ] && echo "$after" | grep -qi auto; then
  echo "WARNING: GPU clock pin is a NO-OP here (perf level still '$after'; read-only"
  echo "         sysfs / unprivileged container). Relying on the 2-pass variance floor."
else
  echo "GPU clocks pinned: '$before' -> '$after'"
fi

for pass in 1 2; do
  for ws in $WORLD_SIZES; do
    csv="$OUTDIR/pass${pass}_ws${ws}.csv"
    echo "=== pass $pass  ws=$ws  -> $csv ==="
    BENCH_WORLD_SIZES="$ws" BENCH_CSV="$csv" \
      python -m benchmark.bench_triton_shmem_ar_rmsnorm
    echo "cooldown ${COOLDOWN}s"; sleep "$COOLDOWN"
  done
done

rocm-smi --resetperfdeterminism >/dev/null 2>&1 || true
rocm-smi --setperflevel auto     >/dev/null 2>&1 || true

echo; echo "===== variance floor (|pass1 - pass2| / min, per config) ====="
python - "$OUTDIR" "$WORLD_SIZES" <<'PY'
import csv, statistics, sys, glob, os
outdir = sys.argv[1]
def load(p):
    d = {}
    if not os.path.exists(p): return d
    for r in csv.DictReader(open(p)):
        lat = float(r["lat_ms"])
        if lat == lat:  # skip NaN
            d[(r["world_size"], r["backend"], r["M"], r["N"])] = lat
    return d
p1 = {}; p2 = {}
for f in glob.glob(f"{outdir}/pass1_ws*.csv"): p1.update(load(f))
for f in glob.glob(f"{outdir}/pass2_ws*.csv"): p2.update(load(f))
by_backend = {}
for k in p1.keys() & p2.keys():
    a, b = p1[k], p2[k]
    delta = abs(a - b) / min(a, b) * 100.0
    by_backend.setdefault(k[1], []).append((delta, k, min(a, b)))
print(f"{'backend':>14} | {'median%':>8} {'p90%':>8} {'max%':>8} | worst config (>5%, ms>=0.5)")
for be, rows in sorted(by_backend.items()):
    ds = sorted(d for d, _, _ in rows)
    med = statistics.median(ds)
    p90 = ds[int(0.9 * (len(ds) - 1))]
    mx = max(ds)
    worst = [ (d,k,m) for d,k,m in rows if d > 5.0 and m >= 0.5 ]
    worst.sort(reverse=True)
    tag = ", ".join(f"ws{k[0]} {k[2]}x{k[3]}={d:.0f}%" for d,k,m in worst[:4]) or "none"
    print(f"{be:>14} | {med:8.1f} {p90:8.1f} {mx:8.1f} | {tag}")
PY
