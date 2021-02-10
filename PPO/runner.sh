#!/bin/bash
for i in {1..25}
do 
	sbatch ppo_batchrun.sh
	sleep 30
done

