#!/bin/bash
#SBATCH -J logistic_admm
#SBATCH -o logistic_admm.%j.out
#SBATCH -e logistic_admm.%j.err
#SBATCH --partition=short-96core
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --time=04:00:00

set -euo pipefail

module purge
module load python/3.9.7
module load mpich

export HYDRA_LAUNCHER=fork

if [ -f ~/mpi_env_mpich/bin/activate ]; then
    source ~/mpi_env_mpich/bin/activate
fi

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "MPI ranks   : $SLURM_NTASKS"
echo "Start time  : $(date)"
echo "--------------------------------------"

mpirun -np 10 python3 logistic_admm_mpi.py

echo "--------------------------------------"
echo "End time    : $(date)"
