#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=edu # The account name for the job.
#SBATCH --job-name=FindOptimalWorker_0 # The job name.
#SBATCH --gres=gpu
#SBATCH --constraint=k80
#SBATCH -c 1
#SBATCH --mem-per-cpu=120gb
module load cuda11.2/toolkit cuda11.2/blas cudnn8.1-cuda11.2
module load anaconda

for N in 0 4 8 16
do
    python main.py --num-of-workers $N
done
# End of script
