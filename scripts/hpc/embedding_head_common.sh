#!/bin/bash

resolve_embedding_dir() {
  local preferred="$1"
  local model_glob="$2"
  shift 2
  local candidates=(
    "$preferred"
    "runs/embeddings/$model_glob"
    "runs/benchmark_context5_hop1/embeddings/$model_glob"
    "outputs/benchmark_context5_hop1/embeddings/$model_glob"
    "outputs/benchmark/embeddings/$model_glob"
  )
  local dir
  for dir in "${candidates[@]}"; do
    if [[ -f "$dir/embeddings.npy" && -f "$dir/manifest.csv" ]]; then
      echo "$dir"
      return 0
    fi
  done

  local found
  found="$(find runs outputs -maxdepth 6 -type f -name embeddings.npy 2>/dev/null | sort || true)"
  echo "ERROR: embeddings not found for pattern: $model_glob" >&2
  echo "Expected a directory containing both embeddings.npy and manifest.csv." >&2
  echo "Tried preferred path: $preferred" >&2
  echo "Found embeddings.npy files under runs/outputs:" >&2
  if [[ -n "$found" ]]; then
    echo "$found" >&2
  else
    echo "  <none>" >&2
  fi
  return 1
}
