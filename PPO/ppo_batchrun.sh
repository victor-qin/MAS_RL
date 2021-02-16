#!/bin/bash
<<<<<<< Updated upstream
#SBATCH -n 1               # Number of cores (-n)
#SBATCH -N 1               # Ensure that all cores are on one Node (-N)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
=======
#SBATCH -n 2               # Number of cores (-n)
#SBATCH -N 1               # Ensure that all cores are on one Node (-N)
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
>>>>>>> Stashed changes
#SBATCH -p shared   # Partition to submit to
#SBATCH --mem=32768           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../batch_logs/myoutput_%j_%a.out  # File to which STDOUT will be written, %j inserts jobid, %a is the job array
#SBATCH -e ../batch_logs/myerrors_%j_%a.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/5.0.1-fasrc02 ; conda info --envs ;
module load python/3.7.7-fasrc01
source activate es100
python3 ppo_avg.py --jobid "${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}"
