import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

import time
import itertools as it
import copy
import xgboost as xgb

from src.modules import add_zeros

#from src.nn_models import FFNN, model_constructor


class XGBGridSearch:
    """Perform a grid search for the best variables"""

    def __init__(self, model, params, n_jobs=-1, cv=5, ):
        """
        Parameters:
        -----------
        
        model : XGBRegressor
        
        params : dictionary 
                with two parameter names as keys with each having a list of multiple values
        
        n_jobs : int
                Number of jobs to run during model training
            
        cv : int
                Number of folds in cross validation
        """
        self.model = model
        self.params = params
        self.n_jobs = n_jobs
        self.cv = cv

    def fit(self, X_train, y_train):
        params = self.params
        model = self.model

        # Get all combinations of parameters
        allNames = sorted(params)
        self.allNames = allNames
        combinations = list(it.product(*(params[Name] for Name in allNames)))

        model_params = model.get_params()

        # Create xgb matrix for data compatibility with xgb
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        
        print(f"Fitting {self.cv} folds for each of {len(combinations)} candidates, totalling {self.cv*len(combinations)} fits.")
        

        result_shape = []
        for name in allNames:
            result_shape.append(len(params[name]))
        
        results = np.ones(len(combinations))*10000
        results_std = np.zeros(len(combinations))

        # Perform grid search
        base_start = time.time()
        times = np.zeros(len(combinations))
        for j, combination in enumerate(combinations):
            print(f"Fitting candidate {j+1} of {len(combinations)}")
            start = time.time()
            # Set parameter values for iteration
            for i, name in enumerate(allNames):
                model_params[name] = combination[i]
                print("   ",name, ":", combination[i], end=" ")
            
            # Perform cross validation
            cvresult = xgb.cv(model_params, xgtrain, num_boost_round=1000, early_stopping_rounds=80,  nfold=self.cv, metrics="rmse")

            # Retrieve results
            test_rmse_mean = cvresult.iloc[-1,2]
            test_rmse_std = cvresult.iloc[-1,3]
            train_rmse_mean = cvresult.iloc[-1,0]
            train_rmse_std = cvresult.iloc[-1,1]
            print(f"\n    Test Result: {test_rmse_mean:.4f}     std: {test_rmse_std:.4f}       n iter: {cvresult.shape[0]}")
            print(f"    Train Result: {train_rmse_mean:.4f}     std: {train_rmse_std:.4f}")

            # Check if this is the best model so far
            if test_rmse_mean < results.min():
                self.best_params = copy.deepcopy(model_params)
                self.best_params['n_estimators'] = cvresult.shape[0]
                self.iterations = cvresult.shape[0]
                
            
            results[j] = test_rmse_mean

            results_std[j] = test_rmse_std

            end = time.time()
            total_time = end-base_start
            total_time_sec = f"{total_time%60:.0f}"
            local_time = end-start
            local_time_sec = f"{local_time%60:.0f}"
            times[j] = local_time

            print(f"\n    elapsed time: {total_time//60:.0f}:{add_zeros(total_time_sec)}, {self.cv} fold time: {local_time//60:.0f}:{add_zeros(local_time_sec)}")
            
            try:
                eta = times[:j+1].mean()*(len(combinations)-(j+1))
                eta_sec = f"{eta%60:2.0f}"
                print(f"    time left: {eta//60:.0f}:{add_zeros(eta_sec)}")
            except:
                pass
            

            print("\n\n")
            
            
        results = results.reshape(result_shape)
        results_std = results_std.reshape(result_shape)
        self.results = results
        self.results_std = results_std

        return results

    @staticmethod
    def plot_results(results,params, fig=None, ax=None, title=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, results.shape[0], figsize = (10,4))
        if title is None:
            title = "Grid Search for XGBRegressor"
        allNames = sorted(params)

        for i in range(results.shape[0]):
            print(i)
            ax[i] = sns.heatmap(results[i], annot=True,
                        yticklabels=params[allNames[1]],
                        xticklabels=params[allNames[2]],
                         fmt='.3g', 
                         cmap=sns.cm.rocket_r)
            ax[i].set_ylabel(allNames[0])
            ax[i].set_xlabel(allNames[1])
            ax[i].set_title(f"{allNames[0]} : {params[allNames[0]][i]}")
        fig.suptitle(title)


    def plot_search(self, fig=None, ax=None, title=None, mark_std=False):
        """Plots results in a heatmap highlighting the best parameter
         combination and all combination achieving results within one std
         of the best one"""

        if fig and ax is None:
            fig, ax = plt.subplots()
        if title is None:
            title = "Grid Search for XGBRegressor"

        results = self.results

        #Plot heatmap of all results
        ax = sns.heatmap(results, annot=True,
                        yticklabels=self.params[self.allNames[0]],
                        xticklabels=self.params[self.allNames[1]],
                         fmt='.3g', 
                         cmap=sns.cm.rocket_r)
        ax.set_ylabel(self.allNames[0])
        ax.set_xlabel(self.allNames[1])
        ax.set_title(title)

        #Find coordinates of the best result
        min_y, min_x = np.nonzero(results == np.amin(results))
        
        #Find coordinates of results within one std of best result
        within_std_y, within_std_x = np.nonzero(results <= np.amin(results)+self.results_std[min_x[0], min_y[0]])
        

        #Mark results within one std of best
        if mark_std:
            for i in range(len(within_std_y)):
                ax.add_patch(Rectangle((within_std_x[i], within_std_y[i]), 1, 1, fill=False, edgecolor='forestgreen', lw=3))
        
        #Mark best result
        for i in range(len(min_y)):
            ax.add_patch(Rectangle((min_x[i], min_y[i]), 1, 1, fill=False, edgecolor='limegreen', lw=3))

        return fig, ax

    @property
    def _best_params(self):
        return self.best_params
    
    @property
    def _best_score(self):
        return self.results.min()

    @property
    def _best_model(self):
        return xgb.XGBRegressor(**self.best_params)

    @property
    def _best_n_iter(self):
        return self.iterations


