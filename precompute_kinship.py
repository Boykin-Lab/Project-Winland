import pandas as pd
from pandas_plink import read_plink
import numpy as np
from numpy import ones, loadtxt, zeros, asarray, errstate
import logging

# === CONSTANTS ===

# Instance labels corresponding to UK biobank datasets.
INSTANCE_LABELS = [
    "24500-1.0", "24502-1.0", "24503-1.0", "24504-1.0",
    "24505-1.0", "24506-1.0", "24507-1.0", "24508-1.0"
]

# Dictionary to rename environmental variables.
ENV_RENAME_DICT = {
    "24500-0.0": "env", "24502-0.0": "24502", "24503-0.0": "24503",
    "24504-0.0": "24504", "24505-0.0": "24505", "24506-0.0": "24506",
    "24507-0.0": "24507", "24508-0.0": "24508"
}  
N = 10_000
logging.basicConfig(level=logging.INFO)
def normalize_env_matrix(E, norm_type="linear_covariance"):
    """
    Normalises the environmental matrix.

    Parameters
    ----------
    E : array
        Matrix of environments
    norm_type : string
        Determines the type of normalization.
        - 'linear_covariance': Normalize so that EE^T has mean of diagonal of ones.
        - 'weighted_covariance': Normalize so that EE^T has diagonal of ones.
        - 'correlation': Normalize so that EE^T is a correlation matrix.

    Returns
    -------
    E : array
        Normalised environments.
    """
    # Removing columns with standard deviation of zero
    E = E[:, E.std(0) > 0]
    E -= E.mean(axis=0)
    E /= E.std(axis=0)
    #z - score calculation 
    # Helper function for linear_covariance normalization

    def linear_covariance(E):
        
        return E * np.sqrt(E.shape[0] / np.sum(E ** 2))

    # Helper function for weighted_covariance normalization
    def weighted_covariance(E):
        return E / np.sqrt((E ** 2).sum(axis=1))[:, np.newaxis]

    # Helper function for correlation normalization
    def correlation(E):
        E -= E.mean(axis=1)[:, np.newaxis]
        return E / np.sqrt((E ** 2).sum(axis=1))[:, np.newaxis]

    normalization_mapping = {
        "linear_covariance": linear_covariance,
        "weighted_covariance": weighted_covariance,
        "correlation": correlation
    }

    # Apply the appropriate normalization
    if norm_type in normalization_mapping:
        E = normalization_mapping[norm_type](E)
    
    return E

def downsample_data(df, fraction=None, n_samples=None):
    # Usage:

    # If you prefer to downsample by a specific number of samples:
    # n_samples_to_sample = 10000
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

# env_df =  pd.read_csv("../data/environment.csv", index_col=0)
# env_df.drop(columns=INSTANCE_LABELS, inplace = False) # removing incomplete instance columns for biobank  

# pheno_data_df =  pd.read_csv("../data/temp2.csv", index_col=0)
# global PHENO_PATH
# PHENO_ID = "height1" #changes column name to pull phenotype data 
# PHENO = "height" #names files after pheno tested
# PHENO_PATH = f'{PHENO}_data'
# pheno_df = pheno_data_df[[PHENO_ID]].dropna(axis=0, inplace=False)

# #Reshape to N x 1 array , using .values since Series
# total_df = pd.merge(pheno_df, env_df, on="eid", how="inner")
# total_df.dropna(axis=0 , inplace=True)

# env_df = total_df.drop(columns=[PHENO_ID], inplace=False)

# E = normalize_env_matrix(env_df.values)  
# env_df_downsampled = downsample_data(pd.DataFrame(E), n_samples=N)

# W = ones((N, 1)) # intercept (covariate matrix)
# pheno_df = pd.DataFrame(total_df[PHENO_ID])
# # print("height shape,", height_df.shape)
# print("Number samples:", N)
# print("Phenotype: ",PHENO)


# pheno_df_downsampled = downsample_data(pheno_df, n_samples=N)
# pheno_df_downsampled.to_csv("10000Sampleout/downsampled_data/10k_height")
# env_df_downsampled.to_csv("10000Sampleout/downsampled_data/10k_env")

bedfile = "../../22418/UKB.merged/UK_F"
try: 
    (bim, fam, G) = read_plink(bedfile, verbose=True)
    
    # for i in range(1,23):
    #     SNP_CHROM_ARR.append((str(i) , bim.chrom == str(i)))
    
except Exception as e:
    logging.error(f"Error reading bed file: {e} ")

print(bim, fam, G)
print(G.shape)

