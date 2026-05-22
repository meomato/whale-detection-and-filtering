#!/bin/bash
#SBATCH --job-name=whale_vox_thresh
#SBATCH --partition=rocky
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1

set -euo pipefail

cd ~/whales/repo
mkdir -p slurm_logs runs/voxaboxen_pilot/threshold_selection

module purge
module load rocky
module load Python/PyTorch_GPU_v2.4

export PYTHONNOUSERSITE=1
source ~/whales/.venv/bin/activate
export PYTHONPATH=~/whales/repo:${PYTHONPATH:-}

python -u -m filtering.finetune.select_voxaboxen_threshold \
  --val-full-results runs/voxaboxen_pilot/fold_01_dataset/project/beats_20ep_fold_01/val_full_results.json \
  --test-full-results runs/voxaboxen_pilot/fold_01_dataset/project/beats_20ep_fold_01/test_full_results.json \
  --output-dir runs/voxaboxen_pilot/threshold_selection \
  --iou 0.5
