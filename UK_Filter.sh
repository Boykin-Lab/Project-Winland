#!/bin/bash

#SBATCH -n 1
#SBATCH -t 20:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email>

module load plink/1.90 
# plink --bfile c1 --merge-list allfiles.txt --make-bed --out UK_merged

#Get all the individuals who self identify as British
# plink --bfile UK_merged --keep ukb9200.2017_8_WinterRetreat.Covars.British.FIDIIDs --exclude Affx.snps.txt --make-bed --out UK_A

#Remove all people with high heterogeneity, cryptic relatedness, and possibly admixed
# plink --bfile UK_A --remove ukb_sqc_v2.wfam.ukbDrops.FIDIIDs --make-bed --out UK_B

#Check for minor allele frequency at each SNP
plink --bfile UKB.merged --maf 0.01 --make-bed --out UK_C
#Check SNPs for hardy-weinberg equilibrium
plink --bfile UK_C --hwe 0.000001 --make-bed --out UK_D
#Check for SNPs that have high missingness 
plink --bfile UK_D --geno 0.01 --make-bed --out UK_E
#Check for individuals with high missingness
plink --bfile UK_E --mind 0.05 --make-bed --out UK_F
#Remove all realted inidividuals, create two separate sets of files - the first contains all remaining SNPS
# plink --bfile UK_F --remove relatedindivs.txt --make-bed --out UK_ALL_REMAINING
#The second is pruned for LD
plink --bfile UK_F --indep-pairwise 100 10 0.1 --make-bed --out UK_pruned
plink --bfile UK_pruned --extract ukb.final.cohort.prune.in --make-bed --out UK_LD
