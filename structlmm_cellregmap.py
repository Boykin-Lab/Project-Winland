import geno_sugar as gs
import geno_sugar.preprocess as prep
import numpy as np
import pandas as pd
import time
import os
import scipy as sp
from numpy import ones, loadtxt, zeros, asarray, errstate
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from limix_core.util.preprocess import gaussianize  
from sklearn.impute import SimpleImputer
from cellregmap import run_association, run_interaction, estimate_betas
from pandas_plink import read_plink
import logging
"""
Dev notes : if you have problems installing the package into python env
 then install directly with python version from env 
 examples : ` /gpfs/home/csomerv1/struct-py/bin/python  -m pip install mymodule `
"""

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
#ranges for snps in the UK_LD.bim 
snp_chr_ranges  = [
# ("01" ,801536 , 249218540) 
# ("02" , 29350 , 242846855),
# ("03" , 72365 , 197840071),
# ("04" , 71566 , 190868396),
# ("05" , 31679 , 180687212),
# ("06" , 203397 , 170889829),
# ("07" , 43748 , 159085413),
# ("08" , 164984 , 146285083),
# ("09" , 45440 , 141066490),
# ("10" , 111955 , 135426536),
# ("11" , 193146 , 134936055),
# ("12" , 208819 , 133810935),
# ("13" , 19121950 , 115096858),
# ("14" , 19336265 , 107250736),
# ("15" , 20163574 , 102383093),
# ("16" , 85667 , 90170095),
# ("17" , 6157 , 81051887),
# ("18" , 82931 , 77998496),
# ("19" , 260970 , 59083268),
# ("20" , 65900 , 62915231),
# ("21" , 14700000 , 48100000) #range doesnt work 
("22" , 17500000 , 28500000)
]

"""
Running 
"""

"""
    bim[(bim.chrom == "1") & (bim.pos < 245379820)] , n = 31947
    bim[(bim.chrom == "2") & (bim.pos < 232498693)]  , n = 30338
    bim[(bim.chrom == "3") & (bim.pos < 197119835)]  , n = 27723
    bim[(bim.chrom == "4") & (bim.pos < 197119835)]  , n = 25892
    bim[(bim.chrom == "5") & (bim.pos < 180695849)]  , n = 25228
    bim[(bim.chrom == "6") & (bim.pos < 170907734)] , n = 28020
    bim[(bim.chrom == "7") & (bim.pos < 159124173)] , n = 22451
    bim[(bim.chrom == "8") & (bim.pos < 146292681)] , n = 21240
    bim[(bim.chrom == "9") & (bim.pos < 141066490)] , n = 17955
    bim[(bim.chrom == "10") & (bim.pos < 135426536)] , n = 20483
    bim[(bim.chrom == "11") & (bim.pos < 134942100)] , n = 20407
    bim[(bim.chrom == "12") & (bim.pos < 133831319)] , n = 19549
    bim[(bim.chrom == "13") & (bim.pos < 115103150)] , n = 14101
    bim[(bim.chrom == "14") & (bim.pos < 107282437)] , n = 13152
    bim[(bim.chrom == "15") & (bim.pos < 102388991)] , n = 12356
    bim[(bim.chrom == "16") & (bim.pos < 90170095)] , n = 13988
    bim[(bim.chrom == "17") & (bim.pos < 81051887)] , n = 13002
    bim[(bim.chrom == "18") & (bim.pos < 78015180)] , n = 12341
    bim[(bim.chrom == "19") & (bim.pos < 59094374)] , n = 10645
    bim[(bim.chrom == "20") & (bim.pos < 62915231)] , n = 10803
    bim[(bim.chrom == "21") & (bim.pos < 48099610)] , n = 6117
    bim[(bim.chrom == "22") & (bim.pos < 51193629)] , n = 6461


"""
    
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


def create_snp_queue(G, bim, Isnp):
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
    G, bim_out = gs.snp_query(G, bim, Isnp)
    # print(bim_out)
    if bim_out.empty:
        logging.error("Bad range, empty range")
        return 

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
    # number of snps in the batch 
    queue = gs.GenoQueue(G, bim_out, batch_size=100, preprocess=preprocess, verbose=True)
    
    return queue
"""
Problems to solve: 

How to split the data to downsample over the entire data set ?
How many jobs will it take?
Can we have the script use the linux(OSCAR) to reduce workload?
How will batch size affect p-values? 

Rather than downsampling within analysis - partion data in the downsample size, loop through

Ensure downsample does not repeat inputs 
"""
def partition_dataframe(df, p):
    if p <= 0:
        raise ValueError("The number of partitions 'p' must be greater than 0")
    
    indices = df.index.tolist()
    partitions = []

    partition_size = len(indices) // p

    for i in range(p):
        if i < p - 1:
            partition, indices = train_test_split(indices, test_size=len(indices) - partition_size, random_state=42 + i)
        else:
            partition = indices

        partitions.append(partition)

    return partitions


def snp_query(G, bim, Isnp):
    r"""
    Parameters
    ----------
    G : (`n_snps`, `n_inds`) array
        Genetic data
    bim : pandas.DataFrame
        Variant annotation
    Isnp : bool array
        Variant filter

    
    Returns
    -------
    G_out : (`n_snps`, `n_inds`) array
        filtered genetic data
    bim_out : dataframe
        filtered variant annotation
    """
    bim_out = bim[Isnp].reset_index(drop=True)
    G_out = G[bim_out.i.values]
    bim_out.i = pd.Series(sp.arange(bim_out.shape[0]), index=bim_out.index)
    return G_out, bim_out

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


# Create a pandas DataFrame (replace this with your own data)
# data = {'Column1': range(1, 101)}
def partition(df, p):
    """
    Input: 
        data : pandas df
        p : number of desired partition 

    """

    # Set the number of batches
    p = 5

    # Calculate the size of each batch
    batch_size = len(df) // p

    # Split the DataFrame into p non-overlapping batches
    batches = [df[i*batch_size:(i+1)*batch_size] for i in range(p)]

    # If there are leftover rows, add them to the last batch
    if len(df) % p != 0:
        batches[-1] = df[(p-1)*batch_size:]

    # Now batches is a list of DataFrames
    for i, batch in enumerate(batches):
        print(f"Batch {i+1}:")
        print(batch)
        print()
    return batches 
# N = 10,000
# P = 1000
# E = N / P  
# i = 0 
# [1, 2, 3, 4]
# for i in range(E):
#     N[P*E, 2(P*E)]

def run_structlmm_test(y, W, E, G, bim, tests):
    # Core testing logic from the original run_structlmm without reading the PLINK bed file

    TESTS = ['interaction', 'association']
    Isnp_arr = []
    for i in range(len(snp_chr_ranges)):
        Isnp_arr.append(gs.is_in(bim, snp_chr_ranges[i]))

    pv_matrix = []
    for Isnp in Isnp_arr:
        num_batch = 0
        queue = create_snp_queue(G, bim, Isnp)
        if queue == None:
            print("! ERROR : QUEUE RANGE INCORRECT!", )
            return
        res = []
        n_analyzed = 0
        t0 = time.time()
        
        for _G, _bim in queue:
            num_batch+=1
            _pv = zeros(_G.shape[1])
            _pv_int = zeros(_G.shape[1])
            # n_analyzed += _G.shape[1]

            downsample_G = downsample_data(pd.DataFrame(_G),  n_samples= N)
            print(" downsample G size ",downsample_G.shape)
            # print("size W: ", W.shape, "size y", y.shape)
            # assert y.shape[0] == W.shape[0]
            hK = compute_kinship(downsample_G)
            print("hK shape: ", hK.shape)
            for snp in range(_G.shape[1]):
                print("SNP #:", snp+1)
                
                
                x = _G.T[snp].reshape(len(_G.T[snp]), 1)
                # print("x:" , x.shape)
                # print("y shape ", y.shape)
               
                down_x = downsample_data(pd.DataFrame(x), n_samples= N)
                print("down x: " , down_x.shape)
                n_analyzed += 1
                # n = down_x.shape[0]
                # print(n)
                # try: 
                # print("pass")
                # y = asarray(y, float)
                # print("asarray", y.shape)
                # y= y.flatten()
                # print("flatten", y.shape)
                # exit()
                # print(W)
                pv = run_interaction(y=y, G=down_x, W=W, E=E, hK=hK)[0]
                with open(str(22)+"pv_snp.txt", "w") as w: 
                    print(f'Interaction test p-value: {pv}')
                    w.write(f'{pv},{snp}\n')
                w.close()
                
                # except Exception as e:
                #     logging.error(f"Error running interaction: {e} ")
                #     return   
                _pv_int[snp] = pv
                # print("success at snp", snp+1)

            print("Number of snps analyzed so far: ", n_analyzed)
            print("Number of batches analyzed so far: ", num_batch)


            if TESTS[0] in tests:
                _bim = _bim.assign(pv_int=pd.Series(_pv_int, index=_bim.index))
            if TESTS[1] in tests:
                _bim = _bim.assign(pv=pd.Series(_pv, index=_bim.index))
            res.append(_bim)
        
        t = time.time() - t0
        if n_analyzed == 0: 
            print("ERROR bad range")
            return 0
        print('%.2f s elapsed' % t)
        print("Total snps analyzed: ", n_analyzed)
        pv_matrix.append(pd.concat(res))
    # Output
    # print(pv_matrix)
    
    # pv_matrix = pd.concat(pv_matrix)
    # pv_matrix.reset_index(inplace=True, drop=True)
    

    return pv_matrix

def run_interaction_analysis(y, W, E):
    # n = 43852
    bedfile = "../../22418/UKB.merged/UK_F"
    try: 
        (bim, fam, G) = read_plink(bedfile, verbose=False)
        
        """

        """
    except Exception as e:
        logging.error(f"Error reading bed file: {e} ")
        return
   
    # Now we call our refactored testing function
    print("Initating interaction test")
    pv_matrix = run_structlmm_test(y, W, E, G, bim, tests="interaction")
    return pv_matrix

def example_interaction_analysis():
    """Illustrative function to demonstrate the interaction analysis using sample data."""
    random = RandomState(1)
    
    # Sample data generation
    n = 30                               # number of samples (cells)
    p = 5                                # number of individuals
    k = 4                                # number of contexts
    y = random.randn(n, 1)               # outcome vector (expression phenotype, one gene only)
    C = random.randn(n, k)               # context matrix (cells by contexts/factors)
    W = ones((n, 1))                     # intercept (covariate matrix)
    hK = random.randn(n, p)              # decomposition of kinship matrix (K = hK @ hK.T)
    g = 1.0 * (random.rand(n, 1) < 0.2)  # SNP vector

    # Interaction test
    try: 
        pv = run_interaction(y=y, G=g, W=W, E=C, hK=hK)[0]
    except Exception as e: 
        logging.error(f"Error running interaction : {e} ") 
        return

    print(f'Interaction test p-value: {pv}')

    # In future, if you wish to include the Association test, uncomment below:
    # pv0 = run_association(y=y, G=g, W=W, E=C, hK=None)[0]
    # print(f'Association test p-value: {pv0}')

    return pv  # Optionally, you can return the p-value if you need it for further analysis


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

def column_normalize(X):
    '''
    column-normalize a matrix, such that for each column vector
    mean=0, std=1
    '''
    X = asarray(X, float)

    with errstate(divide="raise", invalid="raise"):
        return (X - X.mean(0)) / X.std(0)

def main():
    # example_interaction_analysis()  
    # print("Number of Cores : 2") 
    env_df =  pd.read_csv("../data/environment.csv", index_col=0)
    # print("env shape 1: ", env_df.shape)
    pheno_df =  pd.read_csv("../data/temp2.csv", index_col=0).dropna(axis=0, inplace=False)
    # env_df.dropna(axis=0, inplace=True)
    # print("env shape drop na: ", env_df.shape)/
    env_df.drop(columns=INSTANCE_LABELS, inplace = False) # removing incomplete instance columns for biobank  
    # print("env shape dropped cols: ", env_df.shape)
    total_df = pd.merge(pheno_df, env_df, on="eid", how="inner")
    total_df.dropna(axis=0 , inplace=True)

    # print(" merged: ", total_df.shape)

    env_df = total_df.drop(columns=["height1", "height2"], inplace=False)
    # env_df.dropna(axis=0, inplace=True)
    # print("env shape: ", env_df)
    #499845

    E = normalize_env_matrix(env_df.values)  
    # print("E    " , E[0: 5])
    fraction_to_sample = 0.001  # This will sample 10% of the data
    # n = 500
    env_df_downsampled = downsample_data(pd.DataFrame(E), n_samples=N)
    # print("env shape: ", env_df_downsampled.shape)
    w_n = env_df_downsampled.shape[0]
    W = ones(( w_n, 1)) # intercept (covariate matrix)
    height_df = pd.DataFrame(total_df["height1"])
    # print("height shape,", height_df.shape)
    pheno_df_downsampled = downsample_data(height_df, n_samples=N)
    # print("pheno shape", pheno_df_downsampled.shape, "env shape", env_df_downsampled.shape)
    try: 
        res = run_interaction_analysis(y = pheno_df_downsampled, W=W,E=env_df_downsampled)[0]
    except Exception as e:
        logging.error(e) 
        return
    res_df = pd.DataFrame(res)
    print("Export")
    
    if not os.path.exists("chr22out"):
        os.makedirs("chr22out")
    res_df.to_csv(f'chr22out/res_structlmm_{N}samples.csv', index=False)

if __name__ == "__main__":
    main()

