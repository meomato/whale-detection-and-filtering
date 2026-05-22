#!/bin/bash
set -euo pipefail

cd ~/whales/repo
mkdir -p ~/whales/external

module purge
module load rocky
module load CUDA/12.4
module load Python/PyTorch_GPU_v2.4

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/opt/software/python/anaconda/2023_03/envs/tensorflow-gpu2_9_updated/lib:$LD_LIBRARY_PATH
export PIP_NO_CACHE_DIR=1
export FAIRSEQ_COMMIT=920a548ca770fb1a951f7f4289b4d3a0c1bc226f

if [ ! -d ~/whales/animal2vec_venv ]; then
  python -m venv ~/whales/animal2vec_venv --system-site-packages
fi

source ~/whales/animal2vec_venv/bin/activate
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  soundfile==0.13.1 \
  scikit-learn==1.5.2 \
  bitarray==3.4.3 \
  sacrebleu==2.4.3 \
  portalocker==3.1.1 \
  tabulate==0.9.0 \
  regex==2024.11.6 \
  cython==0.29.37 \
  --only-binary=:all:

python -m pip install antlr4-python3-runtime==4.9.3 omegaconf==2.3.0 --no-deps
python -m pip install hydra-core==1.3.2 --no-deps

python - <<'PY'
from pathlib import Path
import site

site_packages = Path(site.getsitepackages()[0])
patch_file = site_packages / "sitecustomize.py"
patch_file.write_text(
    r'''
"""Compatibility patches for legacy fairseq on Python 3.11+."""

import copy
import dataclasses

_original_get_field = dataclasses._get_field


def _fairseq_legacy_get_field(cls, a_name, a_type, default_kw_only):
    try:
        return _original_get_field(cls, a_name, a_type, default_kw_only)
    except ValueError as exc:
        message = str(exc)
        if "mutable default" not in message or "use default_factory" not in message:
            raise
        default = getattr(cls, a_name, dataclasses.MISSING)
        if default is dataclasses.MISSING or isinstance(default, dataclasses.Field):
            raise
        setattr(
            cls,
            a_name,
            dataclasses.field(default_factory=lambda default=default: copy.deepcopy(default)),
        )
        return _original_get_field(cls, a_name, a_type, default_kw_only)


dataclasses._get_field = _fairseq_legacy_get_field
'''.lstrip(),
    encoding="utf-8",
)
print("wrote", patch_file)
PY

if [ ! -d ~/whales/external/fairseq-${FAIRSEQ_COMMIT}/fairseq ]; then
  rm -rf ~/whales/external/fairseq-${FAIRSEQ_COMMIT} ~/whales/external/fairseq-${FAIRSEQ_COMMIT}.zip
  python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve

commit = "920a548ca770fb1a951f7f4289b4d3a0c1bc226f"
url = f"https://github.com/facebookresearch/fairseq/archive/{commit}.zip"
out = Path.home() / "whales" / "external" / f"fairseq-{commit}.zip"
print("downloading", url)
urlretrieve(url, out)
print(out)
PY
  python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

commit = "920a548ca770fb1a951f7f4289b4d3a0c1bc226f"
archive = Path.home() / "whales" / "external" / f"fairseq-{commit}.zip"
target = Path.home() / "whales" / "external"
with ZipFile(archive) as zf:
    zf.extractall(target)
print(target / f"fairseq-{commit}")
PY
fi

python - <<'PY'
import os
import sys
from pathlib import Path

commit = "920a548ca770fb1a951f7f4289b4d3a0c1bc226f"
sys.path.insert(0, str(Path.home() / "whales" / "external" / f"fairseq-{commit}"))
mods = ["torch", "torchaudio", "fairseq", "soundfile", "hydra", "omegaconf"]
for name in mods:
    mod = __import__(name)
    print(name, "OK", getattr(mod, "__version__", ""))
PY

echo "animal2vec environment is ready"
