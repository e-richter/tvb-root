#!/bin/bash -l
#SBATCH --job-name="2node-limitcycle-pymc-2"
#SBATCH --account="ich012"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emilius.richter@fu-berlin.de
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
#SBATCH --output=pymc_data/slurm2.out

source ~/.bashrc
source activate BrainSim-env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

theano-cache purge
stamp=$(date +%s)
export THEANO_FLAGS="base_compiledir=/var/tmp/$stamp/.theano/,compile__timeout=24,compile__wait=20,device=cpu"

python3 /users/erichter/tvb/tvb-root/tvb_contrib/tvb/contrib/inversion/examples/2node-limit_cycle/pymc3/pymc_inference-2.py
