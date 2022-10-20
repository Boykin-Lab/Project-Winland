#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=200G
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mail-user=<email>

module load plink/1.90
plink --bfile ukb<DataIDNum>_c1_b0_v2 --merge-list all.files.txt --make-bed --out UKB.merged
