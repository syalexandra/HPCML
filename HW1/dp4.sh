#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=ys3535
#SBATCH --job-name=dp4
#SBATCH -c 1
#SBATCH --time=1:00
#SBATCH --mem-per-cpu=1gb
python dp4.py 300000000 20