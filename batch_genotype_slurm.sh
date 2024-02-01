#!/bin/bash
#SBATCH -N 1
#SBATCH -n 48 
#SBATCH -t 96:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=charles_somerville@alumni.brown.edu

export PYTHONUNBUFFERED=TRUE
source ~/struct-py/bin/activate
module load python/3.9.0
echo "Batch data Chr22 48 cores"
python batch_genotype_data.py

