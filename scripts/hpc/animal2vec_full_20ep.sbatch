#!/bin/bash
#SBATCH --job-name=whale_a2v_full
#SBATCH --partition=rocky
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "type_e|type_f|type_h"

set -euo pipefail

cd ~/whales/repo
mkdir -p slurm_logs runs/animal2vec_detector

module purge
module load rocky
module load CUDA/12.4
module load Python/PyTorch_GPU_v2.4

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/opt/software/python/anaconda/2023_03/envs/tensorflow-gpu2_9_updated/lib:$LD_LIBRARY_PATH

source ~/whales/animal2vec_venv/bin/activate
export FAIRSEQ_COMMIT=920a548ca770fb1a951f7f4289b4d3a0c1bc226f
export PYTHONPATH=~/whales/repo:~/whales/external/animal2vec:~/whales/external/fairseq-${FAIRSEQ_COMMIT}:${PYTHONPATH:-}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python -u -m filtering.finetune.prepare_window_split \
  --windows outputs/benchmark_context5_hop1/annotations_all/windows.csv \
  --splits outputs/benchmark_context5_hop1_cv/annotations_all/fold_01/splits.csv \
  --output-dir runs/animal2vec_detector/fold_01_windows

python -u -m filtering.finetune.finetune_animal2vec \
  --animal2vec-dir ~/whales/external/animal2vec \
  --checkpoint ~/whales/external/animal2vec/checkpoints/animal2vec_large_pretrained_MeerKAT_240507.pt \
  --audio-dir data/benchmark_audio \
  --annotations outputs/benchmark_context5_hop1/annotations_all/annotations_manifest.csv \
  --train-windows runs/animal2vec_detector/fold_01_windows/train_windows.csv \
  --val-windows runs/animal2vec_detector/fold_01_windows/val_windows.csv \
  --test-windows runs/animal2vec_detector/fold_01_windows/test_windows.csv \
  --output-dir runs/animal2vec_detector/fold_01 \
  --batch-size 4 \
  --epochs 20 \
  --min-epochs 5 \
  --patience 3 \
  --selection-metric event_f1_0_5 \
  --event-merge-gap-s 0.0 \
  --threshold-sweep-steps 101 \
  --encoder-lr 1e-5 \
  --head-lr 1e-4 \
  --num-workers 2
