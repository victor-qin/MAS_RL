#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one Node
#SBATCH -p shared # Partition
#SBATCH --mem=4096           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o batch_logs/myoutput_%j_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e batch_logs/myerrors_%j_%a.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/5.0.1-fasrc01
source activate es100
python3 linear_multiple_avg.py
