#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150GB
#SBATCH --time=2:00:00
#SBATCH --job-name='Ploting'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='Ploting.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'Ploting.py'