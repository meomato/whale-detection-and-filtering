#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 START_INDEX [START_INDEX ...]" >&2
  exit 2
fi

project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

for start_index in "$@"; do
  chunk_name="$(printf 'chunk_%05d' "$start_index")"
  echo "RUN ${chunk_name} start=${start_index}"
  bash "${project_dir}/scripts/run_animal2vec_pretrained_chunk.sh" \
    "$start_index" \
    "$chunk_name" \
    512
done
