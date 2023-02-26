#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=edu # The account name for the job.
#SBATCH --job-name=FindOptimalWorker_0 # The job name.
#SBATCH -c 16 # The number of cpu cores to use.
#SBATCH --mem-per-cpu=120gb # The memory the job will use per cpu core.
module load anaconda
for N in 0 4 8 12 16
do
    python main.py --num-of-workers $N
done
# End of script
