#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
animal2vec_dir="${ANIMAL2VEC_DIR:-$(cd "${project_dir}/../animal2vec" && pwd)}"
venv_path="${ANIMAL2VEC_VENV:-${animal2vec_dir}/.venv}"
checkpoint="${ANIMAL2VEC_CHECKPOINT:-${animal2vec_dir}/checkpoints/animal2vec_large_pretrained_MeerKAT_240507.pt}"
audio_dir="${BENCHMARK_AUDIO_DIR:-${project_dir}/data/benchmark/audio}"

cd "$animal2vec_dir"
source "${venv_path}/bin/activate"

if [[ -n "${ANIMAL2VEC_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${ANIMAL2VEC_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python "${project_dir}/filtering/benchmark/extract_animal2vec.py" \
  --animal2vec-dir "$animal2vec_dir" \
  --checkpoint "$checkpoint" \
  --audio-dir "$audio_dir" \
  --windows "${project_dir}/outputs/benchmark/annotations_all/windows.csv" \
  --output-dir "${project_dir}/outputs/benchmark/embeddings/animal2vec_pretrained_meerkat_all" \
  --model-name animal2vec_pretrained_meerkat \
  --batch-size 8
