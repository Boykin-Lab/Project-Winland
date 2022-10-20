#!/bin/sh
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --time=120:00:00
#SBATCH --mail-user=<email>

module load flashpca/2.0
module load plink/1.90
flashpca_x86-64 --bfile ukb.final.cohort -d 20 --outpc UK_overlaps_pcs.txt