#!/bin/bash

#SBATCH --nodes=12
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --job-name='Shear'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zb2113@nyu.edu
#SBATCH --output='Shear.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'Shear.py'