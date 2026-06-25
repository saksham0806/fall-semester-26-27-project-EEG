#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up CUDA environment variables..."
# Export dynamic link library paths for nvidia pip packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/cufft/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/curand/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/cusolver/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/cusparse/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/nccl/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/nvtx/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/:$(pwd)/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib/

echo "Launching EEG Model training pipeline on GPU..."
.venv/bin/python run_models.py
