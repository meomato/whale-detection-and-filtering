#!/usr/bin/env bash
set -euo pipefail

module purge
module load rocky || true
module load CUDA/12.4 || true
module load Python/Miniconda || true

mkdir -p slurm_logs outputs/finetune

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
python -m pip install -e .
python -m pip install "transformers>=4.51" "accelerate>=1.0" "joblib>=1.4"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
