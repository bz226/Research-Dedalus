#!/bin/bash

#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --job-name='1e80.33Q=0.0028'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='1e80.33Q=0.0028.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python '1e80.33Q=0.0028.py'