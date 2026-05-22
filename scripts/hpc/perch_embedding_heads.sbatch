#!/bin/bash
#SBATCH --job-name=whale_perch_heads
#SBATCH --partition=rocky
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "type_e|type_f|type_h"

set -euo pipefail

cd ~/whales/repo
mkdir -p slurm_logs runs/embedding_heads/perch_v2

module purge
module load rocky
module load Python/PyTorch_GPU_v2.4

export PYTHONNOUSERSITE=1
source ~/whales/.venv/bin/activate
export PYTHONPATH=~/whales/repo:${PYTHONPATH:-}
source scripts/hpc/embedding_head_common.sh

EMB_DIR="$(resolve_embedding_dir \
  "outputs/benchmark_context5_hop1/embeddings/perch_v2_all_audio" \
  "perch_v2_all_audio")"
echo "Using Perch embeddings: ${EMB_DIR}"

for FOLD in 01 02 03 04 05; do
  COMMON_ARGS=(
    --embeddings "${EMB_DIR}/embeddings.npy"
    --embedding-manifest "${EMB_DIR}/manifest.csv"
    --annotations outputs/benchmark_context5_hop1/annotations_all/annotations_manifest.csv
    --splits "outputs/benchmark_context5_hop1_cv/annotations_all/fold_${FOLD}/splits.csv"
    --scenario annotations_all
    --epochs 50
    --min-epochs 8
    --patience 8
    --batch-size 512
    --selection-metric event_f1_0_5
    --event-merge-gap-s 0.0
    --threshold-sweep-steps 101
  )

  python -u -m filtering.finetune.train_embedding_head_detector \
    "${COMMON_ARGS[@]}" \
    --model-name perch_v2_linear \
    --head-kind linear \
    --dropout 0.1 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --output-dir "runs/embedding_heads_cv/perch_v2_linear/annotations_all/fold_${FOLD}"

  python -u -m filtering.finetune.train_embedding_head_detector \
    "${COMMON_ARGS[@]}" \
    --model-name perch_v2_mlp2 \
    --head-kind mlp2 \
    --hidden-dim 256 \
    --dropout 0.2 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --output-dir "runs/embedding_heads_cv/perch_v2_mlp2/annotations_all/fold_${FOLD}"
done
