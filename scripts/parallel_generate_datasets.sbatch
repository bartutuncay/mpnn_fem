#!/bin/bash
#SBATCH --job-name=ground_truth_data
#SBATCH --output=datasets/logs/output_%A_%a.out
#SBATCH --error=datasets/logs/error_%A_%a.err
#SBATCH --array=0-300                        # indices of simulations to generate
#SBATCH --ntasks=1                           # task per array element
#SBATCH --cpus-per-task=2                    # CPUs per task
#SBATCH --mem-per-cpu=8G                     # memory per task
#SBATCH --time=02:00:00                      # max runtime per task

# Load environment (if needed)
module load stack/.2024-06-silent  gcc/12.2.0 python/3.12.8

# Activate virtual environment (if applicable)
source ~/<env>/bin/activate

# Run Python script, passing the array index as the integer argument
python mpnn_fem/datasets_src/datasets_split/beam/cru.py ${SLURM_ARRAY_TASK_ID}
