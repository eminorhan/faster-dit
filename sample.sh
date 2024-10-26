#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=00:10:00
#SBATCH --job-name=sample_dit
#SBATCH --output=sample_dit_%A_%a.out
#SBATCH --array=1

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

TAES=(
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
TAE=${TAES[$SLURM_ARRAY_TASK_ID]}

INCHANNELS=(16 64 256 64 256 1024 256 1024 4096 1024 4096 16384)
INCHANNEL=${INCHANNELS[$SLURM_ARRAY_TASK_ID]}

NUMPATCHES=(256 256 256 64 64 64 16 16 16 4 4 4)
NUMPATCH=${NUMPATCHES[$SLURM_ARRAY_TASK_ID]}

srun python -u sample.py \
	--tae $TAE  \
	--tae_ckpt /scratch/eo41/tae/outputs/${TAE}/${TAE}_checkpoint.pth \
	--model "DiT_XL" \
	--model_ckpt "/scratch/eo41/faster-dit/results/004-DiT_XL/checkpoints/0090000.pt" \
	--in_channels $INCHANNEL \
	--num_patches $NUMPATCH

echo "Done"