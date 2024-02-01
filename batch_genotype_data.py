import geno_sugar as gs
import geno_sugar.preprocess as prep
import numpy as np
import pandas as pd
import time
from os import path, makedirs
from sklearn.impute import SimpleImputer
from numpy import zeros
from pandas_plink import read_plink
import logging


def downsample_data(df, fraction=None, n_samples=None):
    # Usage:

    # If you prefer to downsample by a specific percentage of samples:
    # n_samples_to_sample = .1 10% of samples 
    # env_df_downsampled = downsample_data(env_df, n_samples=n_samples_to_sample)
    # pheno_df_downsampled = downsample_data(pheno_df, n_samples=n_samples_to_sample)

    if fraction:
        return df.sample(frac=fraction)
    elif n_samples:
        return df.sample(n=n_samples)
    else:
        raise ValueError("Either fraction or n_samples must be provided.")


def compute_kinship(X):
    """
    Compute the kinship matrix from genotype data.
    
    Parameters:
    - X : numpy array of shape (N, M)
          Genotype matrix where N is the number of individuals and M is the number of SNPs.

    Returns:
    - K : numpy array of shape (N, N)
          The kinship matrix.
    """
    # Standardize the genotype matrix
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Compute the kinship matrix
    K = (1.0 / X.shape[1]) * np.dot(X_std, X_std.T)
    
    return K

def create_snp_queue(G, bim, Isnp=None):
    """
    Prepares a queue of genotypic data after applying various preprocessing steps.

    Parameters
    ----------
    G : array
        Matrix containing genotype data.
    bim : array
        Information matrix about the SNPs.
    Isnp : array
        Indices of SNPs to query from the bim matrix.

    Returns
    -------
    queue : GenoQueue
        Genotypic data queue after preprocessing.
    """
    
    # Query the specific SNPs using Isnp
    
    G, bim_out = gs.snp_query(G, bim, bim.chrom=="22") #bim columns: chrom - snp - cm - pos - a0 - a1 -i, all strings

    # Define the imputation strategy for missing genotypic data
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant')

    # Define the preprocessing pipeline
    preprocess = prep.compose(
        [
            prep.filter_by_missing(max_miss=0.10),           # Filter SNPs with more than 10% missing data
            prep.impute(imputer),                           # Impute missing genotypic data
            prep.filter_by_maf(min_maf=0.10),               # Filter SNPs with minor allele frequency below 10%
            prep.standardize(),                             # Standardize genotypic data
        ]
    )
    # Create a queue of preprocessed genotypic data
    # batch_size = number of snps in the batch 
    queue = gs.GenoQueue(G, bim_out, batch_size=100, preprocess=preprocess, verbose=True)
    return queue


def main():
    bedfile = "../../22418/UKB.merged/UK_F"
    # try: 
    (bim, fam, G) = read_plink(bedfile, verbose=True)
    # except Exception as e:
        # logging.error(f"Error reading bed file: {e}")
    N = 10_000

    num_batch = 0

    queue = create_snp_queue(G, bim) # creating snp interator 
    if queue == None:
        print("! ERROR : QUEUE RANGE INCORRECT!", )
        exit()
    res = []
    n_analyzed = 0
    t0 = time.time()
    print(G)
    for _G, _bim in queue:
        num_batch+=1
        print("Batch #", num_batch, "starting....")
        print("_G SNP batch shape", _G.shape)
        downsample_G = downsample_data(pd.DataFrame(_G),  n_samples= N) #grab the correct N 
        if (downsample_G.shape[1] == 0):  #we've reached the end of the genome 
            break
        print(" downsample G size ",downsample_G.shape)
        hK = compute_kinship(downsample_G)
        print("hK shape: ", hK.shape)
        outpath = f'{N}Sampleout/geno_data/batch{num_batch}' #whatever path you want to save batches in
        if not path.exists(outpath):
            makedirs(outpath)
        hK.to_csv(f'{outpath}/hKinship{num_batch}.csv', index=False)
        makedirs(f'{outpath}/SNP_ARR{num_batch}')
        for snp in range(_G.shape[1]):
            print("SNP #:", snp+1)
            x = _G.T[snp].reshape(len(_G.T[snp]), 1)
            down_x = downsample_data(pd.DataFrame(x), n_samples= N)
            down_x.to_csv(f'{outpath}/SNP_ARR{num_batch}/{num_batch}_snp{snp}.csv', index=False)
        print("Batch #",num_batch,"completed...")
    t = time.time() - t0
    print('%.2f s elapsed' % t)
    

if __name__ == "__main__":
    main()