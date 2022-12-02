#!/bin/bash -l
#SBATCH --job-name="2node-limitcycle-sbi-2"
#SBATCH --account="ich012"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emilius.richter@fu-berlin.de
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
#SBATCH --output=sbi_data/slurm2.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/.bashrc
source activate BrainSim-env

python3 sbi_inference-2.py
