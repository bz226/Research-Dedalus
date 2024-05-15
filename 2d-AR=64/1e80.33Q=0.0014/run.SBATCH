#!/bin/bash

#SBATCH --nodes=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --job-name='1e80.33Q=0.0014'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='1.2e60.33Q=0.0014.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python '1e80.33Q=0.0014.py'