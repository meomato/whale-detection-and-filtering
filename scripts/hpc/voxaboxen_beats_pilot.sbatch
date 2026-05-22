#!/bin/bash
#SBATCH --job-name=whale_vox_pilot
#SBATCH --partition=rocky
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "type_e|type_f|type_h"

cd ~/whales/repo
mkdir -p slurm_logs runs/voxaboxen_pilot

module purge
module load rocky
module load CUDA/12.4
module load Python/PyTorch_GPU_v2.4

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/opt/software/python/anaconda/2023_03/envs/tensorflow-gpu2_9_updated/lib:$LD_LIBRARY_PATH

source ~/whales/.venv/bin/activate
export PYTHONPATH=~/whales/repo:~/whales/external/voxaboxen:$PYTHONPATH

python -u -m filtering.finetune.prepare_voxaboxen_dataset \
  --audio-dir data/benchmark_audio \
  --annotations outputs/benchmark_context5_hop1/annotations_all/annotations_manifest.csv \
  --splits outputs/benchmark_context5_hop1_cv/annotations_all/fold_01/splits.csv \
  --output-dir runs/voxaboxen_pilot/fold_01_dataset

cd ~/whales/external/voxaboxen

python -u main.py train-model \
  --name beats_pilot_fold_01 \
  --project-config-fp ~/whales/repo/runs/voxaboxen_pilot/fold_01_dataset/project/project_config.yaml \
  --encoder-type beats \
  --beats-checkpoint-fp ~/whales/external/voxaboxen/weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
  --clip-duration 5 \
  --clip-hop 1 \
  --batch-size 2 \
  --lr 1e-5 \
  --n-epochs 1 \
  --min-epochs 0 \
  --patience 3 \
  --unfreeze-encoder-epoch 0 \
  --display-pbar -1 \
  --num-workers 2 \
  --n-map 3 \
  --exists-strategy overwrite
