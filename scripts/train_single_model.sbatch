#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH -A account
#SBATCH --output=training/logs/output_%A.out
#SBATCH --error=training/logs/error_%A.err
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00                     
#SBATCH --gpus=rtx_4090:1

# Load stack
module load stack/.2024-06-silent gcc/12.2.0 python_cuda/3.11.6 eth_proxy

# Run Python script
srun python3 -m mpnn_fem.models.multiscale_tanh_pe