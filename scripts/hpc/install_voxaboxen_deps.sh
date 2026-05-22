#!/bin/bash
set -euo pipefail

cd ~/whales/repo

module purge
module load rocky
module load CUDA/12.4
module load Python/PyTorch_GPU_v2.4

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/opt/software/python/anaconda/2023_03/envs/tensorflow-gpu2_9_updated/lib:${LD_LIBRARY_PATH:-}

source ~/whales/.venv/bin/activate
export PYTHONPATH=~/whales/repo:~/whales/external/voxaboxen:${PYTHONPATH:-}

python -m pip install --upgrade pip wheel setuptools

python -m pip install --only-binary=:all: \
  soundfile==0.13.1 \
  scikit-learn==1.5.2 \
  einops==0.8.1 \
  intervaltree==3.2.1 \
  mir_eval==0.8.2 \
  torchmetrics==1.4.1 \
  librosa==0.10.2.post1 \
  audioread==3.0.1 \
  soxr==0.5.0.post1 \
  pooch==1.8.2 \
  lazy_loader==0.4 \
  msgpack==1.1.0 \
  joblib==1.4.2 \
  numba==0.60.0 \
  llvmlite==0.43.0 \
  sortedcontainers==2.4.0 || true

python -m pip install --only-binary=:all: \
  seaborn==0.13.2 \
  pillow==11.2.1 \
  lightning-utilities==0.14.3 \
  filelock==3.18.0 \
  fsspec==2025.5.1 \
  jinja2==3.1.6 \
  networkx==3.4.2 \
  sympy==1.13.3 \
  threadpoolctl==3.6.0 \
  python-dateutil==2.9.0.post0 \
  pytz==2025.2 \
  tzdata==2025.2 \
  requests==2.32.3 || true

python -m pip install --only-binary=:all: \
  soundfile scikit-learn einops intervaltree mir_eval torchmetrics librosa \
  audioread soxr pooch lazy_loader msgpack joblib numba llvmlite \
  sortedcontainers seaborn pillow lightning-utilities filelock fsspec jinja2 \
  networkx sympy threadpoolctl python-dateutil pytz tzdata requests

python - <<'PY'
mods = [
    "torch", "torchvision", "numpy", "scipy", "sklearn",
    "librosa", "numba", "soxr", "audioread", "pooch", "lazy_loader",
    "msgpack", "joblib", "einops", "intervaltree", "mir_eval",
    "torchmetrics", "seaborn", "PIL", "yaml", "tqdm",
]
failed = []
for name in mods:
    try:
        mod = __import__(name)
        print(name, "OK", getattr(mod, "__version__", ""))
    except Exception as exc:
        print(name, "FAIL", exc)
        failed.append(name)
if failed:
    raise SystemExit("missing imports: " + ", ".join(failed))
print("Voxaboxen deps are ready")
PY

