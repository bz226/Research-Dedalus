#!/bin/bash

#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --job-name='with rotation'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='with rotation.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'with rotation.py'