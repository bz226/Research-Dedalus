#!/bin/bash

#SBATCH --nodes=8
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --job-name='Dry'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='Dry.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'Dry.py'