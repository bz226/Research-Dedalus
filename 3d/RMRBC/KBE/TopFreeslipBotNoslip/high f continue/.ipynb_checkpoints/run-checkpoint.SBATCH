#!/bin/bash

#SBATCH --nodes=10
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --time=48:00:00
#SBATCH --job-name='AR=20'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='AR=20.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'AR=20.py'