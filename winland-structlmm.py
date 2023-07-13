import os
import sys
import time
import pandas as pd
import scipy as sp
import scipy.stats as st
import geno_sugar as gs
import numpy as np
import structlmm
from numpy import ones, concatenate, loadtxt, zeros, random, hstack
from numpy.random import RandomState
from limix_core.util.preprocess import gaussianize  
from sklearn.impute import SimpleImputer
from pandas_plink import read_plink

"""
dev/notes

should be run `python winland-structlmm.py [path]`, 
where the third argument is the directory to the data

Must be run with all data in path provided "data_structlmm/" or by fault looks for example data

issues getting numpy : https://anaconda.org/conda-forge/numpy
dependencies for struct-lmm : https://libraries.io/pypi/struct-lmm/0.3.2/tree
"""

TESTS = ['interaction', 'association']
df_dict = {}

# multiple instances of ukbiobank data can do not have a the same number
# of subjects as some subjects did not return to record results 
instance_labels = ("24500-1.0",
    "24502-1.0",
    "24503-1.0",
    "24504-1.0", 
    "24505-1.0",
    "24506-1.0", 
    "24507-1.0",
    "24508-1.0") 

# renaming variables in environmental data, might be unnecesary but
# norm_env_matrix may error with "-" in names
env_rename_dict = {
    "24500-0.0" : "env",
    "24502-0.0" : "24502",
    "24503-0.0" : "24503",
    "24504-0.0" : "24504",
    "24505-0.0" : "24505",
    "24506-0.0" : "24506",
    "24507-0.0" : "24507",
    "24508-0.0" : "24508",
}

# import_one_pheno_from_csv, norm_env_matrix
# taken from https://github.com/limix/struct-lmm/blob/31656379a343786497771ef6a82f91ed9405c240/struct_lmm/utils/sugar_utils.py#L44
def import_one_pheno_from_csv(pdf, pheno_id, standardize=False):
    """
    Utility to import phenos

    Parameters
    ----------
    *altered : pdf -> final data frame for pheno data after merging across env eid's 
    pfile : str
        csv file name. The csv should have row and col readers.
        See example at http://www.ebi.ac.uk/~casale/example_data/expr.csv.
        The file should contain no missing data.
    pheno_id : array-like
        phenotype to extract.
    standardize : bool (optional)
        if True the phenotype is standardized.
        The default value is False.

    Returns
    -------
    y : (`N`, `1`) array
        phenotype vactor
    """
    # read and extract
    df2 = pdf
    key = df2.columns[0]
    Ip = df2[key] == pheno_id
    del df2[key]
    y = df2[Ip].values.compute().T

    assert not sp.isnan(y).any(), "Contains missing data!"

    if standardize:
        y -= y.mean(0)
        y /= y.std(0)

    return y

def norm_env_matrix(E, norm_type="linear_covariance"):
    """
    Normalises the environmental matrix.

    Parameters
    ----------
    E : array
        matrix of environments
    norm_type : string
        if 'linear_covariance', the environment matrix is normalized in such
        a way that the outer product EE^T has mean of diagonal of ones.
        if 'weighted_covariance', the environment matrix is normalized in such
        a way that the outer product EE^T has diagonal of ones.
        if 'correlation', the environment is normalized in such a way that the 
        outer product EE^T is a correlation matrix (with a diagonal of ones).
    Returns
    -------
    E : array
        normalised environments.
    """
    std = E.std(0)
    E = E[:, std > 0]
    E -= E.mean(0)
    E /= E.std(0)
    if norm_type == "linear_covariance":
        E *= np.sqrt(E.shape[0] / np.sum(E ** 2))
    elif norm_type == "weighted_covariance":
        E /= ((E ** 2).sum(1) ** 0.5)[:, sp.newaxis]
    elif norm_type == "correlation":
        E -= E.mean(1)[:, sp.newaxis]
        E /= ((E ** 2).sum(1) ** 0.5)[:, sp.newaxis]
    return E

def make_out_dir(outfile):
    """
    Util function to make out dir given an out file name.

    Parameters
    ----------
    outfile : str
        output file
    """
    resdir = "/".join(sp.array(outfile.split("/"))[:-1])
    if not os.path.exists(resdir):
        os.makedirs(resdir)

# modified code to incorporate non deprececiated numpy & scipy functions
# taken from https://github.com/limix/struct-lmm/blob/31656379a343786497771ef6a82f91ed9405c240/struct_lmm/lmm/run_structlmm.py
def run_structlmm(snps,
                   bim,
                   pheno,
                   env,
                   covs=None,
                   rhos=None,
                   batch_size=1000,
                   tests=None,
                   unique_variants=False):
    """
    Utility function to run StructLMM

    Parameters
    ----------
    snps : array_like
        snps data
    bim : pandas.DataFrame
        snps annot
    pheno : (`N`, 1) ndarray
        phenotype vector
    env : (`N`, `K`)
          Environmental matrix (indviduals by number of environments)
    covs : (`N`, L) ndarray
        fixed effect design for covariates `N` samples and `L` covariates.
    rhos : list
        list of ``rho`` values.  Note that ``rho = 1-rho`` in the equation described above.
        ``rho=0`` correspond to no persistent effect (only GxE);
        ``rho=1`` corresponds to only persistent effect (no GxE);
        By default, ``rho=[0, 0.1**2, 0.2**2, 0.3**2, 0.4**2, 0.5**2, 0.5, 1.]``
    batch_size : int
        to minimize memory usage the analysis is run in batches.
        The number of variants loaded in a batch
        (loaded into memory at the same time).
    tests : list
        list of tests to perform.
        Each element shoudl be in ['association', 'interation'].
        By default, both tests are considered.
    unique_variants : bool
        if True, only non-repeated genotypes are considered
        The default value is False.

    Returns
    -------
    res : *:class:`pandas.DataFrame`*
        contains pv of joint test, pv of interaction test
        (if no_interaction_test is False) and snp info.
    """
    if covs is None:
        covs = ones((pheno.shape[0], 1))

    if rhos is None:
        rhos = [0.0, 0.1 ** 2, 0.2 ** 2, 0.3 ** 2, 0.4 ** 2, 0.5 ** 2, 0.5, 1.0]

    if tests is None:
        tests = TESTS

    if TESTS[0] in tests:
        slmm_int = structlmm.StructLMM(M=pheno, E = env, W=env, y=covs)

    if TESTS[1] in tests:
        slmm = structlmm.StructLMM(M= pheno, E =env, W=env, y=covs)
        null = slmm.fit(verbose=False)

    # geno preprocessing function
    imputer = SimpleImputer(missing_values=sp.nan, strategy='mean')
    impute = gs.preprocess.impute(imputer)
    standardize = gs.preprocess.standardize()
    preprocess = gs.preprocess.compose([impute, standardize])

    # filtering funciton
    filter = None
    if unique_variants:
        filter = gs.unique_variants

    t0 = time.time()

    # loop on geno
    res = []
    n_analyzed = 0
    # print(snps, bim)
    # N  = 274
    # snp# = 994

    queue = gs.GenoQueue(snps, bim, batch_size=50, preprocess=preprocess)
    for _G, _bim in queue:
        print("running.....")
        print(_G.shape)
        _pv = zeros(_G.shape[0])
        _pv_int = zeros(_G.shape[0])
        for snp in range(_G.shape[0]):
            x = _G[[snp], :].T
           #print("x-shape", x.shape)
            if TESTS[0] in tests:
                # interaction test
                # if (_G.shape[1] <  _G.shape[0]):
                #     covs1 = hstack((covs, x))
                slmm_int.fit(verbose=False)
                _p = slmm_int.score_2dof_inter(x)
                _pv_int[snp] = _p
                # print("inter: " + str(counter))
                # counter+=1
            
            if TESTS[1] in tests:
                # association test
                _p = slmm.score_2dof_assoc(x)
                _pv[snp] = _p
        print("test succeeded....")
        if TESTS[0] in tests:
            _bim = _bim.assign(pv_int=pd.Series(_pv_int, index=_bim.index))

        if TESTS[1] in tests:
            _bim = _bim.assign(pv=pd.Series(_pv, index=_bim.index))

        # add pvalues to _res and append to res
        res.append(_bim)

        n_analyzed += _G.shape[0]
        print('.. analysed %d/%d variants' % (n_analyzed, snps.shape[0]))
    
    print(res)
    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)
    t = time.time() - t0
    print('%.2f s elapsed' % t)
    return res

# toRanks, gaussianize 
# taken from https://github.com/limix/limix-core/blob/master/limix_core/util/preprocess.py#L64
def toRanks(A):
    """
    converts the columns of A to ranks
    AA=sp.zeros_like(A)
    for i in range(A.shape[1]):
        AA[:,i] = st.rankdata(A[:,i])
    AA=sp.array(sp.around(AA),dtype="int")-1
    return AA
    """
    print("—————— RANK —————– \n", A)
    AA = st.rankdata(A, method="ordinal", nan_policy="raise")
    #need to reduce all by one to index into array on zero index going forward
    AA=np.array(np.around(AA)) - 1
    print("—————— AA ranked —————– \n", AA)
    print("———————— END RANK ——————")
    return AA

"""   Project Winland code    """
# private function to test how gaussianize handles inputs and outputs
def mygaussianize(Y):
    """
    Gaussianize X: [samples x phenotypes]
    - each phentoype is converted to ranks and transformed back to normal using the inverse CDF
            YY=toRanks(Y)
            quantiles=(sp.arange(N)+0.5)/N
            gauss = st.norm.isf(quantiles)
            Y_gauss=sp.zeros((N,P))
            for i in range(P):
                Y_gauss[:,i] = gauss[YY[:,i]]
            Y_gauss *= -1
            return Y_gauss
    """
    print("———————————————— gaussianize ———————————————— \n")
    N,P = Y.shape
    shape_str = f"\n n: {str(N)} p: {str(P)}  \n"
    print(shape_str)
    
    # giving each subject phenotype a rank 
    YY = toRanks(Y)
    print ("Rank ----- YY\n")
    # creating quantiles based on how many data subjects 
    quantiles = (np.arange(N)+0.5)/N
    # creating postions for a normalized distribution to perform inverse CDF
    gauss = st.norm.isf(quantiles)
    Y_gauss = zeros((N,P))
    print("gauss \n", gauss)  

    # inverse CDF 
    for i in range(N):
        yy = int(YY[i])
        Y_gauss[i] = gauss[yy]
    Y_gauss *= -1
    print ("———————— Y gauss————————", Y_gauss)
    return Y_gauss

def remove_withdraws(df):
    removal_list = loadtxt("withdraw.txt")
    return df.loc[""]

def make_df_dict(path , li):
    #this function takes in a list of str that are the names of the csv's we want to create dfs for
    #returns - dictionary of str:df where keys are the file names and values are dataframes 
    dict = {} 
    for file in li: 
        filename = file.removeprefix(path)
        print(filename)
        dict[file.removeprefix(path)] = pd.read_csv(file, index_col=0)
    return dict

def summarize_data(dict):
    for key in dict:
        print("----- " + key + " -------")
        print(dict[key].head())
        print("\n")

def clean_data(env_df, pheno_df):
    """
    Parameters
    ----------
    env_df : dataframe 
        Panda dataframe of environemnt variable N X M 
    pheno_df : dataframe
        Pandas dataframe of phenotypes being studied, N x M  
    Returns
    -------
    tuple : cleaned data frames (pheno_df, env_df) 
    """
    env_df.drop(columns=instance_labels, inplace =True) # removing incomplete instance columns for biobank  
    env_df.dropna(axis=0, inplace=True)
    pheno_df.dropna(axis=0, inplace=True)
    
    # Getting data that has a subject in both height and environmental variables  
    env_df.rename(columns=env_rename_dict, inplace = True) # removing special characters from column names 
    total_df = pd.merge(pheno_df, env_df, on="eid", how="inner")
    pheno_df = total_df["height1"].dropna(axis=0, inplace=True)
    env_df = total_df.drop(columns=["height1", "height2"], inplace=False).dropna(axis=0, inplace=True)

    # anaylysis must have matching subjects to perform structLMM 
    if (env_df.shape[0] != pheno_df.shape[0]): print("Variables subjects differ")
    shape_msg = f"Env #:{str(env_df.shape[0])}, Pheno #:{str(pheno_df.shape[0])}"
    print(shape_msg)
    return (pheno_df, env_df)

def init_analysis(pheno_df, env_df, analysis_ready):
    """
    Parameters
    ----------
    pheno_df : dataframe
        Pandas dataframe of phenotypes being studied, N x M  
    env_df : dataframe 
        Panda dataframe of environemnt variable N X M 
    Returns
    -------
    N/A : creates a directory "out/" with structLMM results 
    """
    print("norm env")
    E = norm_env_matrix(env_df.values)    

    pheno_T = pheno_df.T #gaussianize expects a transposed dataframe of trait X subject 
    print("\n ___transposed pheno____ \n", pheno_T)

    pheno = gaussianize(pheno_T.loc['height1'].values[:,None]) 
    print("\n ___gaussianized data____ \n", pheno)
    #pheno = mygaussianize(dfp.loc['gene1'].values[:,None]) 

    if (analysis_ready):
        #file prefix for files .bim, .fam .bed preprocessed genomic data 
        bedfile = "data/UK_LD"
        (bim, fam, G) = read_plink(bedfile) 
        Isnp = gs.is_in(bim, ("22", 17500000, 18000000))
        G, bim = gs.snp_query(G, bim, Isnp)
        print(Isnp, G, bim)
        print("\n _______ \n")
        covs = ones((pheno_df.shape[0], 1))
        
        # run analysis with struct lmm
        snp_preproc = {"max_miss": 0.01, "min_maf": 0.02}
        # res = run_structlmm(
        # G, bim, pheno, env_df, covs=covs, batch_size=100
        # )
        print("Beginning structlmm")

        # slmm = structlmm.StructLMM(M= pheno, E =E, W=E, y=covs)
        # slmm.fit(verbose=False)
        # TODO: what is x supposed to be? https://github.com/limix/struct-lmm/blob/master/struct_lmm/test/test_structlmm.py
        # pv =  slmm.score_2dof_assoc(G)
        # print(pv)
        # slmm = StructLMM(y, M, E, W=E)
        # slmm.fit(verbose=False)
        
        res = run_structlmm(
        G, bim, pheno, E, covs=covs, batch_size=142,
        unique_variants=False) #y, M, E

        print("Export")
        if not os.path.exists("out"):
            os.makedirs("out")
        res.to_csv("out/res_structlmm.csv", index=False)

def example_analysis():
    print("———————————————— Example ———————————————— \n")
    # import genotype file
    bedfile = 'data/example_data/chrom22_subsample20_maf0.10'
    (bim, fam, G) = read_plink(bedfile)
    
    # subsample snps
    Isnp = gs.is_in(bim, ('22', 17500000, 18000000))
    G, bim = gs.snp_query(G, bim, Isnp)
    print("======== Bim file ======== \n", 
        Isnp, G, bim)
   
    dfp = pd.read_csv('data/example_data/expr.csv', index_col=0) # load phenotype file
    print("======== gene1 expr ======== \n", 
        dfp.loc['gene1'].values[:, None])
    
    # gaussianize() expects data to be free of nan values and 
    # transposed such that the matrix is M X N
    #                subjects 
    #             ————————————
    # phenotype1 , # , # , # 
    # phenotype2 , # , # , #   
    pheno = gaussianize(dfp.loc['gene1'].values[:, None])
    print("======== Pheno ======== \n")

    # load environment file and normalize
    envfile = 'data/example_data/env.txt'
    E = loadtxt(envfile)
    E = norm_env_matrix(E)

    # mean as fixed effect
    covs = ones((E.shape[0], 1))

    # run analysis with struct lmm
    print("======== Beginning struct ========")
    print("G data: ",G)
    print(E.shape, pheno.shape, bim.shape, covs.shape)
    res = run_structlmm(G, bim, pheno, E, covs=covs, batch_size=100, unique_variants=True)

    # export
    print('Export')
    if not os.path.exists('out'): os.makedirs('out')
    res.to_csv('out/res_structlmm.csv', index=False)

def random_analysis():
    random = RandomState(1)
    n = 20
    k = 4
    y = random.randn(n, 1)
    E = random.randn(n, k)
    M = ones((n, 1))
    x = 1.0 * (random.rand(n, 1) < 0.2)

    lmm = structlmm.StructLMM(y, M, E)
    lmm.fit(verbose=False)
    # Association test
    pv = lmm.score_2dof_assoc(x)
    assert(pv == 0.8470017313426488)
    # Association test
    pv, rho = lmm.score_2dof_assoc(x, return_rho=True)
    assert(pv == 0.8470017313426488)
    assert(rho == 0)
    M = concatenate([M, x], axis=1)
    lmm = structlmm.StructLMM(y, M, E)
    lmm.fit(verbose=False)
    # Interaction test
    pv = lmm.score_2dof_inter(x)
    assert(pv == 0.6781100453132024)

def main():
    args = sys.argv
    numargs = len(args)
    if numargs < 3: SystemError("usage: `python winland-structlmm.py <'research' | 'random' | 'pseudo'> [...path]`")
    type_analysis = args[2]
    #if using psuedo or research there must be a file path selected
    print("""The Winland Project can run 3 analysis: random, psuedo, research
            random : uses random data generated by the Random module
            psuedo : uses psuedo data from LMM documentation
            research : uses data from real UKBiobank variables""")    
    if type_analysis != "research":
        print("start analysis ........")
        if type_analysis == "random" : random_analysis()
        elif type_analysis == "pseudo" : example_analysis()
        return
    elif numargs < 4: SystemError("using research analysis but did not provide a path to retrieve data")
    """
    Biobank data should be formated 
              var1 | var2 | var3
             ———————————————————— 
        eid |
    """
    path = args[4]

    files = (f"{path}phenotype.csv",
        f"{path}environment.csv",
        f"{path}struct_vars.csv", 
        f"{path}temp2.csv")
    df_dict = make_df_dict(path , files)
    # renaming/dropping ukbiobank columns since formating can get messy 
    # and we only care about the first instance of data recorded
    env_df = df_dict["environment.csv"]
    height_df = df_dict["temp2.csv"]
    (pheno_df, env_df) = clean_data(pheno_df, height_df)
    # running our own analysis with phenotype and environmental variables 
    init_analysis(pheno_df=pheno_df, env_df=env_df, analysis_ready=True)

if __name__ == "__main__":
    main()
