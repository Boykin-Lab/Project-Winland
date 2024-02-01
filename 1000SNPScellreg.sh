#!/bin/bash

#SBATCH -n 20
#SBATCH -t 72:30:00
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=charles_somerville@alumni.brown.edu

export PYTHONUNBUFFERED=TRUE
source ~/struct-py/bin/activate
module load python/3.9.0
echo "1000SNPS 22 chroms"
echo "20 Cores"
python structlmm_cellregmap.py






