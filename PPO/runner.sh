#!/bin/bash
for i in {1..20}
do 
	sbatch ppo_batchrun.sh
	sleep 30
done

