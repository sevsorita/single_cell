from inspect import FullArgSpec
from tkinter.messagebox import RETRY
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


import re
import os
import xgboost as xgb


class Dataset:
    """Used for easy loading and handling of scRNA datasets in /data/severs/ directory
    
    Arguments: 
    ----------
    subset : str, which celltype to use
    source_name : str, which dataset to use. Either `BluePrint` or `NeoLetExe`
    nans : whether to include columns with nans
    timepoint : NOT YET implemented which timepoint to use in NeoLetExe data.
    """
    
    def __init__(self, subset="Cancer", source_name = "BluePrint", nans=False, timepoint=None, data=None) -> None:

        dirpath = "/data/severs/" + source_name + "/"
        source_name = "".join(re.findall('([A-Z])', source_name)) 

        if not nans:
            path = dirpath + subset + "_clean_" + source_name + ".pkl"
            drop_nans = False
            if not os.path.exists(path):
                path = dirpath + subset+ "_" + source_name + ".pkl"
                drop_nans = True
        else:
            path = dirpath + subset + "_" +  source_name + ".pkl"
        
        self.df = pd.read_pickle(path)
        
        if drop_nans:
            self.df = self.df.dropna(axis=1)
        
        if "patient_number" not in self.df.columns:
            self.df["patient_number"] = pd.read_csv(dirpath+subset+"_patientnumber_"+source_name+".csv", header=None)[1].values
    
    
    def set_XY(self, target="ESR1", subsampling=0, **kwargs):
        """Set the input and target for the model saved in self.X and self.Y. 
        
        Arguments:
        ----------
        target : str, the gene which will be predicted
        subsampling : int in [0,1,2]
                      if `0` no subsampling is done. `1` removes cells with absolute expression
                      less than 0.5. `2` subsamples the cells with abs expression less
                      than 0.5 and includes some to increase the dataset size by 25%.   
        kwargs : passed on to balanced_datasampling if 
        """

        if subsampling == 1:
            self.df = self.df[abs(self.df[target])>0.5]
        elif subsampling == 2:
            self.df = self.balanced_sampling(self.df, target, **kwargs)

        self.target = target

    @property
    def DMatrix(self):
        return xgb.DMatrix(self.X, self.Y)

    @property
    def X(self):
        return self.df.drop([self.target, "patient_number"], axis=1)
    
    @property
    def Y(self):
        return self.df[self.target]

    @property
    def patient_number(self):
        return self.df["patient_number"]

    def __getitem__(self, val : str):
        if val in self.df.columns:
            return self.df[val]
        else:
            raise ValueError("Not a valid key, key must be a gene or `patient_number`")


    @staticmethod
    def balanced_sampling(df,
                          target, 
                          dataincrease = 0.25, 
                          priority="expression", 
                          exp_limit=0.5,
                          random_state=10) -> pd.DataFrame:
        """ 
        Balances data by elimininating most cells with target-expression in [-exp_limit, exp_limit].
        The sample is stratified by patient number to try to get the balance as close as possible
        to the original balance, and by expression to get as a many different values as possible.
        A perfect balance is not possible and therefore one must prioritize one or the other.

        Arguments:
        ----------
        target : str, the target gene to stratify by
        dataincrease : float, the percentage to increase the differentiated datasize by.
        priority : str, what to prioritize in balancing either `expression` or `patient_number`.
        exp_limit : threshold for differentially expressed cells
        random_state : set random state for reproducability

        Returns: pandas DataFrame with balanced data. %
        """
        np.random.seed(random_state)
        inc_pns = df[abs(df[target])>exp_limit].patient_number.copy()
        exc_pns = df[abs(df[target])<=exp_limit][["ESR1", "patient_number"]].copy()
        exc_pns["ESR1"] = pd.cut(exc_pns.ESR1,bins=np.linspace(-exp_limit,exp_limit,6), labels=np.arange(5))
        n_old_rows = inc_pns.shape[0]
        target_b = df.patient_number.value_counts() / df.shape[0]
        n_new_rows = int(inc_pns.shape[0]*dataincrease)
        filler = np.empty(n_new_rows)
        filler.fill(np.nan)
        inc_pns = inc_pns.append(pd.Series(filler))
        missing_pns = list(set(df.patient_number.unique()) - set(inc_pns.unique()))
        
        balances = np.zeros((n_new_rows, len(df.patient_number.unique())))
        exp_strat = np.zeros(5)
        for i in range(n_old_rows, inc_pns.shape[0]):
            current_b = (inc_pns.value_counts() / inc_pns.shape[0]).copy()

            for p in missing_pns:
                if p not in current_b.index:
                    current_b = current_b.append(pd.Series({p:0}))
        
            inbalance = target_b - current_b
            balances[i-n_old_rows] = inbalance.sort_index()

            if priority=="expression":
                bin = np.argmin(exp_strat)
                for pn in inbalance.sort_values(ascending=False).index:
                    pn_df = exc_pns[(exc_pns.patient_number==pn) & (exc_pns.ESR1==bin)]
                    if pn_df.shape[0]>0:
                        exp_strat[bin]+=1
                        break
            
            elif priority=="patient_number":
                new_pn = inbalance.index[np.argmax(inbalance)][0]
                for bin in np.argsort(exp_strat):
                    pn_df = exc_pns[(exc_pns.patient_number==new_pn) & (exc_pns.ESR1==bin)]
                    if pn_df.shape[0]>0:
                        exp_strat[bin]+=1
                        break
            else:
                raise ValueError("priority must be either `expression` or `patient_number`.")
            index = np.random.randint(pn_df.shape[0])
            inc_pns.at[i-n_old_rows] = pn_df.patient_number[index]
            inc_pns.rename(index={i-n_old_rows:pn_df.index[index]},inplace=True)
            exc_pns.drop(pn_df.index[index], inplace=True)
        
        return df.loc[inc_pns.index].copy()

    
    def train_test_split(self, test_size=0.2, stratify="patient_number", random_state=None):
        """
        Split the data in training and testing and stratify by either `patient_number` or 
        `target`.
        Arguments:
        ----------
        test_size : float, share of data in the test set
        stratify : which variable to stratify by. Either `patient_number` or `target` i.e 
                   target gene.
        random_state : int, for reproducability

        Return:
        -------

        """

        if stratify == "patient_number":
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            for train, test in sss.split(self.df, self.patient_number):
                return self.X.iloc[:,train], self.X.iloc[:,test], self.Y[train], self.Y[test]
        
        elif stratify==self.target or stratify == "target":
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            y_groups = pd.cut(self.Y, bins=np.linspace(self.Y.min()-0.0001, self.Y.max()+0.0001, 6), labels=np.arange(5))
            
            for train, test in sss.split(self.df, y_groups):
                self.split = {"X_train" : self.X.iloc[train],
                            "X_test" : self.X.iloc[test], 
                            "y_train":self.Y[train],
                            "y_test": self.Y[test],
                            "patient_number_train" : self.patient_number[train], 
                            "patient_number_test":self.patient_number[test],
                            "test_ind" : test,
                            "train_ind" : train}
                return self.X.iloc[train], self.X.iloc[test], self.Y[train], self.Y[test]

        else:
            pass

    def cross_validation():
        #TODO: implement cross validation on stratified target gene split
        #TODO: implemnt cv across patients
        pass