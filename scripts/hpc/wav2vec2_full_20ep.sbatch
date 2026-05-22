#!/bin/bash
#SBATCH --job-name=whale_w2v_full
#SBATCH --partition=rocky
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "type_e|type_f|type_h"

set -euo pipefail

cd ~/whales/repo
mkdir -p slurm_logs runs/wav2vec2_detector

module purge
module load rocky
module load CUDA/12.4
module load Python/PyTorch_GPU_v2.4

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/opt/software/python/anaconda/2023_03/envs/tensorflow-gpu2_9_updated/lib:$LD_LIBRARY_PATH

source ~/whales/.venv/bin/activate
export PYTHONPATH=~/whales/repo:${PYTHONPATH:-}

python -u -m filtering.finetune.prepare_window_split \
  --windows outputs/benchmark_context5_hop1/annotations_all/windows.csv \
  --splits outputs/benchmark_context5_hop1_cv/annotations_all/fold_01/splits.csv \
  --output-dir runs/wav2vec2_detector/fold_01_windows

python -u -m filtering.finetune.finetune_wav2vec2 \
  --model-id facebook/wav2vec2-base \
  --audio-dir data/benchmark_audio \
  --annotations outputs/benchmark_context5_hop1/annotations_all/annotations_manifest.csv \
  --train-windows runs/wav2vec2_detector/fold_01_windows/train_windows.csv \
  --val-windows runs/wav2vec2_detector/fold_01_windows/val_windows.csv \
  --test-windows runs/wav2vec2_detector/fold_01_windows/test_windows.csv \
  --output-dir runs/wav2vec2_detector/fold_01 \
  --batch-size 4 \
  --epochs 20 \
  --min-epochs 5 \
  --patience 4 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --selection-metric event_f1_0_5 \
  --event-merge-gap-s 0.0 \
  --threshold-sweep-steps 101 \
  --gradient-checkpointing \
  --num-workers 2
