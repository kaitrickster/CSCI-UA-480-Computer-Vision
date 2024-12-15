#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --nstasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=100MB
#SBATCH --output=slurm_output.out
#SBATCH --job-name=assign1
#SBATCH --mail-type=END
#SBATCH --mail-user=kl3199@nyu.edu
