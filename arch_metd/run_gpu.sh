#!/usr/bin/env bash
# METD phase 5 — run the nested-CV training on the GPU workstation.
#
#   ssh <you>@10.30.161.74
#   tmux new -s metd            # the full run outlives an SSH drop only inside tmux
#   cd ~/eeg && ./arch_metd/run_gpu.sh
#
# Run this ON THE HOST, not inside a container — it starts the container itself.
# Everything it produces lands in arch_metd/results/ on the host, owned by you.
#
#   --check   preflight only, start nothing
#   --smoke   stop after the smoke run (~2 min), skip the real one
#
# What must be on the box first (from the laptop, ~500 MB total, NOT the 5 GB corpora):
#   rsync -avP --exclude .venv --exclude .git --exclude datasets ./ <you>@<box>:~/eeg/
#   rsync -avP datasets/metd/ <you>@<box>:~/eeg/datasets/metd/
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE=eeg-project:torch
DATA=datasets/metd/metd.npz
DATA_MD5=896709049685aeb349efc456edce00d4
MODE=${1:-full}

fail() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$DATA" ] || fail "$DATA is missing. Copy it up:
    rsync -avP datasets/metd/ <you>@<box>:~/eeg/datasets/metd/"

command -v docker >/dev/null || fail "docker not found on this host"
nvidia-smi -L >/dev/null 2>&1 || fail "no GPU visible on the host — check nvidia-smi"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -z "${TMUX:-}" ] && [ "$MODE" = full ]; then
  echo "WARNING: not inside tmux. The full run takes hours and an SSH drop will kill it."
  echo "         Ctrl-C now, then: tmux new -s metd"
  sleep 5
fi

if [ "$MODE" = --check ]; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    && echo "preflight OK — image $IMAGE present, rerun without --check to train" \
    || echo "preflight OK — image $IMAGE not built yet, the real run will build it (~10 min)"
  exit 0
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE from the repo Dockerfile (one time, ~10 min) ..."
  docker build -t "$IMAGE" .
fi

mkdir -p arch_metd/results
if [ "$MODE" = --smoke ]; then SMOKE_ONLY=1; else SMOKE_ONLY=0; fi

# --user keeps results owned by you rather than root (GPU_Run.md gotcha);
# --shm-size=8g or DataLoader workers die with a bus error.
docker run --rm --gpus all --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e SMOKE_ONLY="$SMOKE_ONLY" \
  -v "$PWD:/workspace/eeg" -w /workspace/eeg \
  "$IMAGE" bash -c "
set -euo pipefail
echo '=== container: GPU check ==='
python -c 'import torch; assert torch.cuda.is_available(), \"container sees no GPU\"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

echo '=== data provenance ==='
# Not a byte checksum: two boxes rebuilding from the same corpora can differ in gate
# decisions at floating-point boundaries and in row order, and results computed against
# different arrays are not comparable in ways no summary statistic reveals. Print the
# fingerprint so a mismatch with the laptop is one line, not a forensic exercise.
python -m arch_metd.data.build --fingerprint
echo 'if that fingerprint differs from the laptop, the arrays differ — say so before'
echo 'comparing any result against previous numbers.'

echo '=== gates (seconds, no training) ==='
python -m arch_metd.models.mtdann      # GRL sign, anchor gate, masked loss, max-norm
python -m arch_metd.train.folds        # no subject on both sides, EEGMAT held out
python -m arch_metd.eval.aggregate --selfcheck-only

echo '=== smoke run ==='
python -u -m arch_metd.train.loop --smoke

if [ \"\$SMOKE_ONLY\" = 1 ]; then echo 'smoke only — stopping here'; exit 0; fi

echo '=== full nested CV: main + the three §8.3 controls ==='
# The controls cost the same as the main run and are what make it interpretable — a
# result with no method-off comparison is unfalsifiable. Do them in one session rather
# than discovering later that a second GPU trip is needed.
python -u -m arch_metd.train.loop --all-modes 2>&1 | tee arch_metd/results/train.log

echo '=== cross-corpus transfer (§6.2) ==='
# Runs AFTER the CV on purpose. The §6.5 gates are read off the CV output, and a failed
# stress head makes every transfer number unanswerable — better to see that in the same
# log than to read a transfer table that means nothing.
python -u -m arch_metd.train.loop --transfer 2>&1 | tee -a arch_metd/results/train.log
"

echo
echo "done. results in arch_metd/results/:"
ls -la arch_metd/results/
echo
echo "then, back on the laptop:"
echo "  python -m arch_metd.eval.aggregate --mode main   # and lambda0 / randbb / permuted"
echo "  python -m arch_metd.eval.probes                  # §8.1 probes + the pre-registered gate"
echo "  python -m arch_metd.eval.controls                # §8.4 cross-head"
echo "  python -m arch_metd.eval.report                  # assembles arch_metd/REPORT.md"
echo
echo "pull them back from the laptop with:"
echo "  rsync -avP <you>@<box>:~/eeg/arch_metd/results/ arch_metd/results/"
