#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=00:05:00
#SBATCH --job-name=train_dit
#SBATCH --output=train_dit_%A_%a.out
#SBATCH --array=1

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

MODELS=(
	tae_patch16_vocab16_px256
	tae_patch16_vocab64_px256
	tae_patch16_vocab256_px256
	tae_patch32_vocab64_px256
	tae_patch32_vocab256_px256
	tae_patch32_vocab1024_px256
	tae_patch64_vocab256_px256
	tae_patch64_vocab1024_px256
	tae_patch64_vocab4096_px256
	tae_patch128_vocab1024_px256
	tae_patch128_vocab4096_px256
	tae_patch128_vocab16384_px256
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

srun python -u train.py \
	--train_data_path "/scratch/projects/lakelab/data_frames/imagenet-1k-processed/tae_patch16_vocab64_px256/imagenet_1k_val_tae_patch16_vocab64_px256.pth" \
	--model "DiT-S/8" \
	--compile

echo "Done"