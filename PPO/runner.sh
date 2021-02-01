#!/bin/bash
for i in {1..10}
do 
	sbatch ppo_batchrun.sh
	sleep 45
done

