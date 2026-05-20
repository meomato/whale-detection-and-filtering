#!/usr/bin/env bash
#SBATCH --job-name=whale_gpu_check
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

set -euo pipefail

mkdir -p slurm_logs
module purge
module load rocky || true
module load CUDA/12.4 || true
module load Python/Miniconda || true

source .venv/bin/activate
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
