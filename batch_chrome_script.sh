#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email>@brown.edu
#SBATCH --array=1-22


<path of gfetch>  <DataIDNum> -c$SLURM_ARRAY_TASK_ID 

