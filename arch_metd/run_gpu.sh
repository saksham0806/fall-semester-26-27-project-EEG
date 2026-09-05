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

# a truncated rsync of a 485 MB file fails much later and much more confusingly
echo "checking $DATA ..."
GOT=$(md5sum "$DATA" | cut -d' ' -f1)
[ "$GOT" = "$DATA_MD5" ] || fail "$DATA is corrupt or stale.
    expected $DATA_MD5
    got      $GOT
  Re-copy it, or rebuild it on the laptop with: python -m arch_metd.data.build"

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

echo '=== gates (seconds, no training) ==='
python -m arch_metd.models.mtdann      # GRL sign, anchor gate, masked loss, max-norm
python -m arch_metd.train.folds        # no subject on both sides, EEGMAT held out

echo '=== smoke run ==='
python -u -m arch_metd.train.loop --smoke

if [ \"\$SMOKE_ONLY\" = 1 ]; then echo 'smoke only — stopping here'; exit 0; fi

echo '=== full nested CV ==='
python -u -m arch_metd.train.loop 2>&1 | tee arch_metd/results/train.log
"

echo
echo "done. results in arch_metd/results/:"
ls -la arch_metd/results/
echo
echo "pull them back from the laptop with:"
echo "  rsync -avP <you>@<box>:~/eeg/arch_metd/results/ arch_metd/results/"
