from inspect import FullArgSpec
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
#import statsmodels.stats.multitest as multi
import scipy
import re
import os

def load_cancer():
    df = pd.read_pickle("/data/severs/rna_scaled_cancer.pkl")
    df = transpose_set_ind(df)
    return df


def transpose_set_ind(df):
    """Restructures the data correctly"""
    df = df.rename(columns={'Unnamed: 0': 'Genes'})
    df = df.set_index(["Genes"])
    df = df.T
    return df


def plot_expression_dist(genes, df):
    """Plots distribution of selected genes

    :param genes: list of strings
    :param df: dataframe with the data
    """
    for gene in genes:
        try:
            y = df[gene].sort_values()
            x = np.arange(0,y.shape[0])
            plt.plot(x, y, label=gene)
        except ValueError:
            print(f"Gene {gene} not in dataframe")

    plt.legend()
    plt.ylabel("Expression level")
    plt.xlabel("Cells sorted by expression for each gene")
    plt.title(f"Distribution of gene expression, n={y.shape[0]}")



class VariableSelector:
    """
    Class to store correlaton, p-values and number-of-non-NaNs and select input variables
    
    Paramaters:
    -----------
    df : pd.DataFrame
        dataframe of cell subtype and gene expressions
    """

    def __init__(self, cell_type, dataset_filepath=None) -> None:        

        self.cell_type = cell_type

        self.corr_dfs = {}

        #check if dataset exists
        if dataset_filepath is None:
            dataset_filepath = "/data/severs/reduced_data_sets/" + cell_type + ".pkl"
        if not os.path.isfile(dataset_filepath):
                raise IndexError(f"Dataset for {cell_type} does not exist at {dataset_filepath}")

        self.df = pd.read_pickle(dataset_filepath)



    def select_variables(self, target_gene, diff_lim, verbose=0, nan_limit=0.2, alpha=None, select_n=1000, save_corr=False, positive_exp=False):
        """
        Perform variable selection on the dataset for the chosen target gene(s)

        Parameters:
        -----------
        target_gene : str or list of str
            must be gene in dataset
        verbose : int or boolean

        return : tuple of variables and alpha or list of tuples on the form
                (list of variables, alpha : float)
        """

        if type(target_gene) == str:
            target_gene = [target_gene]

        selected_variables = []
        alphas = []

        if positive_exp:
            s  = "_positive"
        else:
            s = ""


        for i, target in enumerate(target_gene):
            path = "/data/severs/correlation_p_values/" + self.cell_type +"_"+ target +"_correlation_expdiff_"+str(diff_lim)+s+".pkl"
            
            if os.path.isfile(path):
                with open(path, 'rb') as handle:
                        self.corr_dfs[target] = pickle.load(handle)

            else:
                self.corr_dfs[target] = compute_p_value(self.df, target, verbose=verbose, positive_exp=positive_exp)
                if save_corr:
                    with open(path, 'wb') as handle:
                        pickle.dump(self.corr_dfs[target], handle)

            df_ = remove_high_nan(self.corr_dfs[target], limit=nan_limit)
            df_, alpha_local = select_by_alpha(df_, alpha, select_n, verbose)
            selected_variables.append(df_.index)
            alphas.append(alpha_local)
            

        if len(selected_variables) == 1:
            return selected_variables[0], alphas[0]
        else:
            return_array = []
            for variables, alpa in zip(selected_variables, alphas):
                return_array.append((variables, alpha))
            return return_array


    def compute_corr(self, target, diff_lim=0.5, verbose=0, save_corr=True, positive_exp=False):
        """
        Load or compute the correlation dataframe for the selected target gene
        """
        if type(target) == str:
            target = [target]
        if positive_exp:
            s  = "_positive"
        else:
            s = ""
        for target in target:
            path = "/data/severs/correlation_p_values/" + self.cell_type +"_"+ target +"_correlation_expdiff_"+str(diff_lim)+s+".pkl"

            if os.path.isfile(path):
                with open(path, 'rb') as handle:
                        self.corr_dfs[target] = pickle.load(handle)
            else:
                self.corr_dfs[target] = compute_p_value(self.df, target, diff_lim=diff_lim, verbose=verbose, positive_exp=positive_exp)
                if save_corr:
                        with open(path, 'wb') as handle:
                            pickle.dump(self.corr_dfs[target], handle)

    def extract_data(self, target_gene, diff_lim=0.5, verbose=0, nan_limit=0.2, alpha=None, select_n=1000, save_corr=False, positive_exp=False):
            """
            Get inputs and outputs based criteria in parameters. All parameters are passed to select_variables

            return : X (DataFrame), Y (Series), alpha (float)
            """
            selected, alpha = self.select_variables(target_gene, diff_lim, verbose, nan_limit, alpha, select_n, save_corr)

            df = self.df

            X = df[selected]
            Y = df[target_gene]
            
            if positive_exp:
                    X = X[Y>diff_lim]
                    Y = Y[Y>diff_lim]
            else:
                    X = X[Y.abs()>diff_lim]
                    Y = Y[Y.abs()>diff_lim]
            
            return X, Y, alpha

def filter(Y):
    return Y > 0.5

lambda y : y > 0.5


def compute_p_value(df, target, diff_lim=0.5, fdr=True, verbose=1, positive_exp=False):
    """
    Compute correlation between target and all other genes that have less NaNs than
    nan_limit, the corresponding FDR p-values are computed afterwards.
    """
 
    target_col = df[target]
    df = df.drop(target, axis=1)

    if positive_exp:
        df = df[target_col>diff_lim]
        target_col = target_col[target_col>diff_lim]
    else: 
        df = df[target_col.abs()>diff_lim]
        target_col = target_col[target_col.abs()>diff_lim]

    df = df.loc[:, df.isnull().sum() < df.shape[0]-3]

    n_col = len(df.columns)
    non_nan_count = np.empty(n_col)

    if verbose:
        prog_bar = tqdm
    else: 
        prog_bar = lambda x: x


    if verbose: print("Counting NaNs:")
    # Check if it can be computeted efficiently
    if target_col.isnull().sum() == 0:
        for i in prog_bar(range(n_col)):
            non_nan_count[i] = df.shape[0]-df[df.columns[i]].isnull().sum()     # This can be sped up with simply df.isnull().sum()
            
    # Slow if not
    else:
        for i in prog_bar(range(n_col)):
            non_nan_count[i] = count_non_nans(target_col, df[df.columns[i]])
            

    correlation = np.empty(n_col)
    p_value = np.zeros(n_col)

    if verbose: print("Computing correlation and p-value:")
    # Compute correlation and p-value
    for i in prog_bar(range(n_col)):
        if non_nan_count[i] < 3:
            correlation[i] = 0
            p_value[i] = 1
        else:
            correlation[i] = target_col.corr(df[df.columns[i]])
            t = correlation[i] * np.sqrt(non_nan_count[i]-2) / np.sqrt(1 - correlation[i]**2) # t-test
            p_value[i] = scipy.stats.t.sf(abs(t), non_nan_count[i]-2)*2
            if p_value[i] == float("nan"):
                print(correlation[i])
                print(t)
                print(non_nan_count[i])
    
    #if fdr:
    #    p_value = multi.multipletests(p_value, method="fdr_bh")[1]

    p_value[np.where(p_value>1)[0]] = 1.0

    corr_df = pd.DataFrame()
    corr_df["correlation"] = correlation
    corr_df["p-value"] = p_value
    corr_df["n_non_NaNs"] = non_nan_count
    corr_df.index = df.columns

    # Test
    
    if corr_df.isnull().sum().sum():
        print("Dropping NANS!!!")
        print("correlation: ", corr_df["correlation"].isnull().sum())
        print("P-values:    ", corr_df["p-value"].isnull().sum())
        print("n_non_NaNs:  ", corr_df["n_non_NaNs"].isnull().sum())
    
        #raise ValueError(f"There are {corr_df.isnull().sum().sum()} NaNs in the correlation / p-value dataframe, this happens if there is an error in the computation of p-value or correlation. Contact Severin if this error is encountered.")

    return corr_df


def remove_high_nan(corr_df, limit=0.2):
    return corr_df[corr_df["n_non_NaNs"] > (1-limit) * corr_df["n_non_NaNs"].max()]


def select_by_alpha(corr_df, set_alpha=None, n=1000, verbose=1):
    """
    
    """

    if set_alpha:
        return corr_df[corr_df["p-value"] < set_alpha], set_alpha

    else:
        alphas = corr_df.sort_values("p-value")["p-value"]
        set_alpha = alphas[n]
    
        try:
            s = str(set_alpha).split("e-")[1]
            zeros = int(s)
        except IndexError:
            s = str(set_alpha)
            zeros = len(re.search('\d+\.(0*)', s).group(1))
        # This part rounds alpha to and selects the closest rounding that achieves
        #  n_inputs within 10% of the selected n
        i = 0
        while i < 5:
            try_alpha = round(set_alpha, zeros+i)
            n_selected_genes = corr_df[corr_df["p-value"] < try_alpha].shape[0]
            if abs(n_selected_genes - n) < 0.1*n:
                if verbose:print(f"Selected {n_selected_genes} input genes with alpha = {try_alpha}")
                return corr_df[corr_df["p-value"] < try_alpha], try_alpha
            else:
                i +=1
        
        return corr_df[corr_df["p-value"] < set_alpha], set_alpha


        raise ValueError("WARNING: It didn't work out, you gotta do something more with the code over here!")


def compute_correlation(df, target, fdr=False, efficient=False):
    """Reduce input data to variables with highest correlation.

    Parameters
    ----------
    df : pandas.DataFrame
        dataset to be reduced

    target : string
        name of variable of interest in df

    returns : correlation with target, p_values, number of non-nans
    """

    # Compute correlation
    column_1 = df[target]
    correlation = np.zeros(len(df.columns))
    p_value = np.ones(len(df.columns))
    n = np.zeros(len(df.columns))
    if efficient:
        for i in tqdm(range(len(df.columns))):
            n[i] = count_non_nans(column_1, df[df.columns[i]])
            
            if df.columns[i] == target:
                correlation[i] = 1.0
                p_value[i] = 0.0
                continue

            if n[i] > 0.8*column_1.shape[0]:
                correlation[i] = column_1.corr(df[df.columns[i]])
                # Compute p-value
                t = correlation[i] * np.sqrt(n[i]-2) / np.sqrt(1 - correlation[i]**2) # t-test
                p_value[i] = scipy.stats.t.sf(abs(t), n[i]-2)*2

    if not efficient:
        for i in tqdm(range(len(df.columns))):
                n[i] = count_non_nans(column_1, df[df.columns[i]])
                
                if df.columns[i] == target:
                    correlation[i] = 1.0
                    p_value[i] = 0.0
                    continue
                        
                else:
                    correlation[i] = column_1.corr(df[df.columns[i]])
                    # Compute p-value
                    t = correlation[i] * np.sqrt(n[i]-2) / np.sqrt(1 - correlation[i]**2) # t-test
                    p_value[i] = scipy.stats.t.sf(abs(t), n[i]-2)*2


    #if fdr:
    #    p_value = multi.multipletests(p_value*2, method="fdr_bh")[1]

    return correlation, p_value, n


def create_corr_df(correlation, n, p_values, columns, p_values_title="FDR p-values"):
    """Create a data frame from arrays from the function compute_correlation"""
    df = pd.DataFrame()
    df = df.append(pd.Series(correlation, name = "Correlation"))
    df = df.append(pd.Series(p_values, name = p_values_title))
    df = df.append(pd.Series(n, name = "n_non_NaNs"))
    df.columns = columns
    df = df.T
    return df


def variable_selector(corr_df, selected_variables=1000, select_by_corr=True, nan_share=0.8, p_limit = 0.05):
    """Select variables from a create_corr_df dataframe by eliminating genes
     with high rates of NaNs and those with high p-values
    """

    corr_df = corr_df[corr_df['n_non_NaNs'] > nan_share*corr_df['n_non_NaNs'].max()]

    corr_df = corr_df[corr_df['FDR p-values'] < p_limit]
    corr_df["Correlation"] = np.abs(corr_df["Correlation"])
    if select_by_corr:
        input_variables = corr_df.sort_values("Correlation").index[-(selected_variables+1):-1]
    else:
        input_variables = corr_df.sort_values("FDR p-values").index[:selected_variables+1]

    return input_variables



def complete_variable_selector(df,
                                target, 
                                fdr=True,
                                selected_variables=1000,
                                select_by_corr=True,
                                nan_share=0.8,
                                p_limit = 0.05,
                                save_corr_to_pickle=False, 
                                save_input_vars=False):

    correlation, p_values, n = compute_correlation(df, target, fdr=fdr, efficient=True)

    corr_df = create_corr_df(correlation, n, p_values, df.columns)

    if type(save_corr_to_pickle) == str:
        corr_df.to_pickle(save_corr_to_pickle)
    
    input_variables = variable_selector(corr_df,
                                        selected_variables=selected_variables,
                                        select_by_corr=select_by_corr,
                                        nan_share=nan_share,
                                        p_limit = p_limit)
    if type(save_input_vars) == str:
        with open(save_input_vars, 'wb') as handle:
                pickle.dump(input_variables, handle)
    
    return input_variables



def count_non_nans(x, y):
    """Counts every instance where corresponding elements in x and y are numbers"""
    import math
    n= x.shape[0]
    for i, j in zip(x, y):
        if math.isnan(i) or math.isnan(j):
            n-=1
    return n


def root_mse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


