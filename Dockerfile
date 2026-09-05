# Image for the GPU workstation (sjt516scope064). Full workflow in GPU_Run.md.
#
#   docker build -t eeg-project:torch .
#   docker run -it --rm --gpus all --shm-size=8g -v $PWD:/workspace/eeg -w /workspace/eeg eeg-project:torch bash
#
# Code and data are bind-mounted at run time, not COPYed — the image is the environment only,
# so rebuilding never costs a 672 MB dataset copy.
#
# This covers the torch track (DL_Without_Features/, distilled/). Keras work needs the NGC
# TensorFlow base instead; see GPU_Run.md section 4. Do not merge the two.
FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace/eeg
COPY requirements.txt .

# torch/torchaudio/tensorflow are filtered out deliberately: this base ships the CUDA torch
# build, while requirements.txt pins the CPU wheel for local use. Installing it unfiltered
# downgrades torch and every "GPU" run silently completes on CPU.
RUN pip install --no-cache-dir $(grep -vE '^\s*(#|$)' requirements.txt \
      | grep -vE '^(torch|torchaudio|tensorflow)$' | tr '\n' ' ')

# braindecode hard-imports torchaudio; the base image does not always ship it.
RUN python -c "import torchaudio" || pip install --no-cache-dir torchaudio==2.10.0

# Fails the build rather than a 3-hour run if the base image lost its CUDA torch.
RUN python -c "import torch, braindecode; assert torch.version.cuda, 'CPU torch in image'"

CMD ["bash"]
