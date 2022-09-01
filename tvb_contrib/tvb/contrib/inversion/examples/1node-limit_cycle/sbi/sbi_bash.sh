#!/bin/bash -l
#SBATCH --job-name="1node-limitcycle-sbi"
#SBATCH --account="ich012"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emilius.richter@fu-berlin.de
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
#SBATCH --output=sbi_data/slurm.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/.bashrc
source activate BrainSim-env

python3 sbi_inference.py
