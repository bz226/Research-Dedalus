#!/bin/bash

#SBATCH --nodes=16
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --job-name='AR=32'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='AR=32.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'AR=32.py'