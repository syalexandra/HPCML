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
module load anaconda
python main.py --num-of-workers 4
python main.py --num-of-workers 4 --batch-normalize FALSE


# End of script
