#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --job-name='plotm'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='plotm.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'plot multifolder.py'