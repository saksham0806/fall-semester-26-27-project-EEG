#!/usr/bin/env bash
# METD — run the whole pipeline locally in the .venv. No Docker, no GPU.
#
#   ./arch_metd/run_local.sh            gates, CV (4 modes), transfer (4 arms), analysis
#   ./arch_metd/run_local.sh --check    preflight only, start nothing
#   ./arch_metd/run_local.sh --smoke    smoke CV + smoke transfer (~6 min), then stop
#   ./arch_metd/run_local.sh --cv       gates + CV only
#   ./arch_metd/run_local.sh --transfer transfer only (CV must already be done)
#   ./arch_metd/run_local.sh --analyse  no training: aggregate, probes, controls, report
#
# ⚠️ Local torch is CPU-only, so the full run is HOURS, not the ~12 minutes it takes on
# the GPU box. Extrapolated from smoke timings: roughly 6-9 h for everything. Run it under
# tmux, or use --cv and --transfer to split it across sessions.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATA=datasets/metd/metd.npz
LOG=arch_metd/results/run_local.log
MODE=${1:-full}

fail() { echo "ERROR: $*" >&2; exit 1; }
stage() { echo; echo "===== $* ====="; date +"      %H:%M:%S"; }

[ -x "$PY" ] || fail "$PY not found — this script expects the project .venv"
[ -f "$DATA" ] || fail "$DATA missing. Build it: $PY -m arch_metd.data.build"

stage "preflight"
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
$PY -m arch_metd.data.build --fingerprint
$PY -c "
from arch_metd.data.build import load
from arch_metd.train.folds import split_hash
import sklearn
print('split hash', split_hash(load()), '| sklearn', sklearn.__version__)"
free -h | awk 'NR<=2'
avail=$(free -m | awk 'NR==2{print $7}')
[ "$avail" -gt 2500 ] || echo "WARNING: only ${avail} MB available; the build peaks near 1.5 GB."

# results/ is overwritten in place. Run A is snapshotted separately on purpose — two
# splits are two valid draws, and anything that survives both is real.
if [ -d arch_metd/results_split_A ]; then
  echo "note: previous draw preserved in arch_metd/results_split_A/ (see its MANIFEST.md)"
  KEPT=$($PY -c "
import json,pathlib
p=pathlib.Path('arch_metd/results_split_A/cv.json')
print(json.loads(p.read_text()).get('fingerprint','?') if p.exists() else '?')")
  HERE=$($PY -c "
from arch_metd.data.build import fingerprint, load
print(fingerprint(load()))")
  if [ "$KEPT" != "$HERE" ]; then
    cat <<MSG

⚠️  ARRAY MISMATCH — the robustness comparison would be confounded.
      kept draw (Run A) was built on : $KEPT
      this machine's arrays are      : $HERE
    Run A came from the workstation, whose rebuild differs from this one at
    floating-point gate boundaries and in row order. Running here changes BOTH the
    arrays and the split, so a result that moves between the two draws cannot be
    attributed to the split. Two clean options:
      - copy the workstation's metd.npz down, so both draws share arrays; or
      - treat this as an independent third run and do not diff it against Run A.
MSG
  fi
fi

cat <<MSG

CPU-only timing, extrapolated from smoke runs (the GPU box does all of this in ~12 min):
  gates + smoke        ~6 min
  CV, 4 modes          ~4-5 h      (5 folds x 4 modes, early stopping ~30-70 epochs)
  transfer, 16 runs    ~2-4 h
  analysis             ~15 min     (probes dominate)
Use --cv and --transfer to split it across sessions, and run under tmux.
MSG

if [ "$MODE" = --check ]; then echo; echo "preflight OK — rerun without --check"; exit 0; fi

mkdir -p arch_metd/results
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

if [ "$MODE" != --analyse ]; then
  stage "gates (seconds, no training)"
  $PY -m arch_metd.contract                      # paths, array contract, 5 seeded bugs
  $PY -m arch_metd.models.mtdann                 # GRL sign, anchor gate, masked loss, pos_weight
  $PY -m arch_metd.train.folds                   # row-order invariance, no subject on both sides
  $PY -m arch_metd.eval.aggregate --selfcheck-only
fi

case "$MODE" in
  --smoke)
    stage "smoke CV";        $PY -u -m arch_metd.train.loop --smoke --all-modes
    stage "smoke transfer";  $PY -u -m arch_metd.train.loop --smoke --transfer
    echo; echo "smoke done — artifacts are tagged _smoke and do not touch the real ones"
    exit 0 ;;
  --cv)
    stage "CV: main + 3 controls"; $PY -u -m arch_metd.train.loop --all-modes ;;
  --transfer)
    stage "transfer: 4 directions x 4 arms"; $PY -u -m arch_metd.train.loop --transfer ;;
  --analyse) ;;
  full)
    stage "CV: main + 3 controls"; $PY -u -m arch_metd.train.loop --all-modes
    # transfer AFTER the CV: the §6.5 gates are read off the CV output, and a stress head
    # that still does not discriminate makes every transfer number unanswerable
    stage "transfer: 4 directions x 4 arms"; $PY -u -m arch_metd.train.loop --transfer ;;
  *) fail "unknown option $MODE" ;;
esac

if [ "$MODE" = --cv ] || [ "$MODE" = --transfer ]; then
  echo; echo "done. run with --analyse once both CV and transfer exist."; exit 0
fi

stage "analysis"
for m in main lambda0_locked lambda0 randbb permuted; do
  [ -f "arch_metd/results/cv$([ "$m" = main ] || echo "_$m").json" ] || continue
  $PY -u -m arch_metd.eval.aggregate --mode "$m"
done
$PY -u -m arch_metd.eval.controls          # §8.4 cross-head
$PY -u -m arch_metd.eval.probes            # §8.1 probes + the pre-registered gate (~12 min)
$PY -u -m arch_metd.eval.report

stage "done"
echo "read arch_metd/REPORT.md"
echo "compare against the kept draw: diff <(jq -S . arch_metd/results/aggregate.json) \\"
echo "                                    <(jq -S . arch_metd/results_split_A/aggregate.json)"
