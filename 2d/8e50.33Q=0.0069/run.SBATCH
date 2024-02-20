#!/bin/bash

#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --job-name='8e50.33Q=0.0069'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='8e50.33Q=0.0069.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python '8e50.33Q=0.0069.py'