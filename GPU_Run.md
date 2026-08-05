# GPU_Run.md — running this repo on the VIT workstation (Docker + CUDA)

`RUN.md` is the command reference for a local `.venv`. This file is only about the extra
steps for the allocated GPU box, where the rule is **run inside a Docker container**.

Box: `sjt516scope064` / `10.30.161.74:22`, user `akshat_23bkt0003`, Ubuntu 24.04,
24 cores / 64 GB RAM / 16 GB GPU / 1 TB. Allocation expires 2027-07-30.

**What actually needs the GPU.** Local torch is `2.13.0+cpu`, so the only real blocker is
the raw-EEG track — `DL_Without_Features/benchmark.py` and `distill.py` have never run
outside `--smoke`. Everything classical (phase1/phase2, all3 SVMs, hierarchical) runs fine
on CPU and does **not** justify a container. Do not port work here that CPU already
finishes in 10 minutes.

---

## 1. One-time setup

### 1.1 NGC account and login

Create the account at <https://ngc.nvidia.com>, then **Setup → Generate API Key**. Save the
key; it is shown once.

```bash
ssh akshat_23bkt0003@10.30.161.74
docker login nvcr.io
# Username: $oauthtoken      <- literally this string, not your email
# Password: <your NGC API key>
```

`$oauthtoken` is the username for every NGC pull. Getting this wrong is the usual first failure.

### 1.2 Pull the image

The raw-EEG track is pure torch, so take the NGC PyTorch container — it already ships
CUDA, cuDNN and a matching torch build.

```bash
docker pull nvcr.io/nvidia/pytorch:25.12-py3     # check the catalog for the current YY.MM tag
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.12-py3 \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

That last line must print `True`. If it prints `False`, the container has no GPU — stop and
fix that before anything else (`nvidia-smi` on the host, then `docker run --rm --gpus all
nvcr.io/nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi`).

### 1.3 Copy code and data up

`datasets/` is gitignored, so nothing about the data comes from git. The raw-EEG track only
reads `datasets/all3/` (672 MB) — do **not** rsync the full 14 GB of raw corpora.

```bash
# from the laptop, in the project root
rsync -avP --exclude '.venv' --exclude '.git' --exclude 'datasets' \
  ./ akshat_23bkt0003@10.30.161.74:~/eeg/
rsync -avP datasets/all3/ akshat_23bkt0003@10.30.161.74:~/eeg/datasets/all3/
```

Add `features_v2/` (46 MB) only if you plan to run the feature-based all3 scripts up here.

---

## 2. Start the container

```bash
docker run -it --name eeg --gpus all \
  --shm-size=8g \
  -v ~/eeg:/workspace/eeg \
  -w /workspace/eeg \
  nvcr.io/nvidia/pytorch:25.12-py3 bash
```

- `--shm-size=8g` — torch DataLoader workers use `/dev/shm`; the 64 MB default causes
  "bus error / DataLoader worker killed".
- `-v ~/eeg:/workspace/eeg` — the code and `Results/` live on the host, so a destroyed
  container loses nothing.
- `--name eeg` (not `--rm`) — needed so the container survives to be committed in §5.

### 2.1 Install the deps, but not torch

The image's torch is the CUDA build. Installing this repo's `requirements.txt` verbatim
**replaces it with the CPU wheel** and silently reverts you to a CPU run.

```bash
pip install $(grep -vE '^\s*(#|$)' requirements.txt | grep -vE '^(torch|torchaudio|tensorflow)$' | tr '\n' ' ')
python -c "import torchaudio" || pip install torchaudio   # braindecode hard-imports it
python -c "import torch, braindecode; print(torch.__version__, torch.cuda.is_available(), braindecode.__version__)"
```

If that `pip install torchaudio` pulls a build that breaks `import torch`, install it from
the CUDA index matching the container's torch (`--index-url https://download.pytorch.org/whl/cu126`).

Sanity-check before spending GPU hours:

```bash
python -m DL_Without_Features.data      # data invariants, ~10 s
python -m DL_Without_Features.core      # 9 asserts including the KD mask
python -u DL_Without_Features/benchmark.py --smoke
```

---

## 3. The real runs

Long runs must survive the SSH session dying. Use `tmux` **on the host**, outside Docker,
then `docker exec` into the container:

```bash
# host
tmux new -s bench
docker exec -it eeg bash
# container
cd /workspace/eeg
python -u DL_Without_Features/benchmark.py > bench.log 2>&1
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t bench`. Watch with `tail -f bench.log`
from a second shell — do not pipe the run itself through `tail`, it buffers until exit.

Then the distillation track, which reads `benchmark.json` to pick its backbone:

```bash
python -u DL_Without_Features/distill.py > distill.log 2>&1
```

**Check the anchor row before believing any of it.** `EEGNet [segment]` should land near
**0.4306**. Far off means the harness is wrong; above ~0.85 means the subject-grouped split
broke and you are reading leakage, not a result.

Smoke output goes to `Results/DL_Without_Features/_smoke/` and real output to
`Results/DL_Without_Features/` — that separation is deliberate, prediction filenames collide
otherwise. Do not merge them.

---

## 4. TensorFlow work (optional)

`run.sh`, `run_models.py` and `scripts/all3_deep.py` are Keras. They need a different image —
do not put TF and CUDA-torch in one container.

```bash
docker pull nvcr.io/nvidia/tensorflow:25.02-tf2-py3
docker run -it --name eeg-tf --gpus all --shm-size=8g \
  -v ~/eeg:/workspace/eeg -w /workspace/eeg \
  nvcr.io/nvidia/tensorflow:25.02-tf2-py3 bash
# inside:
pip install $(grep -vE '^\s*(#|$)' requirements.txt | grep -vE '^(torch|torchaudio|tensorflow|braindecode)$' | tr '\n' ' ')
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -u run_models.py      # NOT ./run.sh — see below
```

`./run.sh` only exists to point `LD_LIBRARY_PATH` at the `.venv`'s nvidia pip packages.
Inside an NGC container CUDA is already on the path and there is no `.venv`, so run
`run_models.py` directly. `run.sh` will fail here.

This also needs the full `datasets/combined_data/` (681 MB), which the §1.3 rsync skipped.

---

## 5. Commit the container so setup isn't repeated

```bash
# host, container still running or stopped
docker commit eeg eeg-project:torch
docker images | grep eeg-project
```

Reuse it:

```bash
docker run -it --rm --gpus all --shm-size=8g \
  -v ~/eeg:/workspace/eeg -w /workspace/eeg eeg-project:torch bash
```

Commit only captures the **image filesystem**, not the `-v` mount — code and results stay on
the host either way. Recommit after any `pip install` you want to keep.

---

## 6. Pull results back

```bash
# from the laptop
rsync -avP akshat_23bkt0003@10.30.161.74:~/eeg/Results/DL_Without_Features/ Results/DL_Without_Features/
rsync -avP akshat_23bkt0003@10.30.161.74:~/eeg/bench.log ./
```

Then update the accuracy×probe grid in `DL_Without_Features/README.md` and append a dated
entry to `dl_plan.md` — that file is a running log, not just a spec.

---

## Gotchas

- **`$oauthtoken` is the NGC username.** Not your email, not your name.
- **Never `pip install -r requirements.txt` unfiltered in an NGC image.** It downgrades torch
  to the CPU wheel and you get a silent CPU run at full GPU-queue cost.
- **`--shm-size=8g`**, or DataLoader workers die with a bus error.
- **Files written in the container are root-owned on the host.** Either add
  `--user $(id -u):$(id -g)` to `docker run`, or `sudo chown -R $USER ~/eeg` afterwards
  before rsyncing back.
- **`-u` on every python invocation.** Redirected stdout is block-buffered otherwise and a
  multi-hour run is unmonitorable.
- **16 GB GPU here vs 6 GB on the laptop** — the batch sizes in this repo (`BATCH_SIZE=128`)
  are tuned for the small card and are not a constraint anymore. Raising them changes
  results; if you do, say so, don't compare across batch sizes silently.
- **`benchmark.py` writes JSON after every model**, so a killed run keeps finished rows.
  Restarting re-runs everything though — there is no resume.
- **Allocation expires 2027-07-30.** Anything living only on that box is temporary; results
  belong in git, data does not.
