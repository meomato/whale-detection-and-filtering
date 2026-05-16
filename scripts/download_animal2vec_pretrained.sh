#!/usr/bin/env bash
set -u

target_size=5024620014
url="https://edmond.mpg.de/api/access/datafile/253220"
project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
animal2vec_dir="${ANIMAL2VEC_DIR:-$(cd "${project_dir}/../animal2vec" && pwd)}"
output_file="${ANIMAL2VEC_CHECKPOINT:-${animal2vec_dir}/checkpoints/animal2vec_large_pretrained_MeerKAT_240507.pt}"

mkdir -p "$(dirname "$output_file")"

attempt=1
while [[ ! -f "$output_file" || "$(stat -c%s "$output_file")" -lt "$target_size" ]]; do
  size="$(stat -c%s "$output_file" 2>/dev/null || echo 0)"
  echo "attempt=$attempt size=$size target=$target_size"

  curl \
    -L \
    --retry 5 \
    --retry-delay 5 \
    --connect-timeout 60 \
    --speed-time 120 \
    --speed-limit 1024 \
    -C - \
    -o "$output_file" \
    "$url" || true

  new_size="$(stat -c%s "$output_file" 2>/dev/null || echo 0)"
  if [[ "$new_size" == "$size" ]]; then
    echo "no progress, sleeping"
    sleep 10
  fi

  attempt=$((attempt + 1))
  if [[ "$attempt" -gt 80 ]]; then
    echo "too many attempts"
    exit 2
  fi
done

stat -c "%n %s" "$output_file"
