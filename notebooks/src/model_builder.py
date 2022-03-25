from os import posix_fallocate
import pandas as pd
import numpy as np
import numpy.ma as ma
import os.path
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import shap
from tqdm.auto import tqdm
import time


import src.data_handling as dh
import src.modules as modules
from src.model_selection import XGBGridSearch


class CellLevelModel:
        """Build models for to predict expression of each selected gene in a cell_type
        
        Provides functions to perform cross validation for each gene of interest or
        build predictive models for each"""

        def __init__(self, cell_type : str, genes : list) -> None:

                #check if valid cell_type
                if cell_type not in ['B_cell', 'EC', 'Myeloid', 'Cancer','Fibroblast', 'T_cell', 'DC', 'Mast', 'Cancer_2mnd', 'Cancer_4mnd', 'Cancer_baseline']:
                        raise ValueError("cell_type not valid, must be in ['B_cell', 'EC', 'Myeloid', 'Cancer','Fibroblast', 'T_cell', 'DC', 'Mast']")

                self.cell_type = cell_type
                self.genes = genes

                self.models = {}  

                self.VS = dh.VariableSelector(self.cell_type)  

                self.tuned = False
                self.data_built = False
                self.shap_analysis_complete = False
        

        def __getitem__(self, name: str):
                """Use to retrieve models from the class
                
                Parameters:
                -----------
                name : str
                        A gene that has been defined in init's 'genes'"""
                try:
                        return self.models[name]
                except:
                        KeyError("Model is not built for this gene")


        def build_model(self, model_params, genes=None, cv=5, early_stopping_rounds=80, build_single_models=True, seed=10):
                """Build models for every gene in genes
                
                Parameters: 
                -----------
                model_params : dict
                        dict of XGBRegressor parameters
                genes : list of str
                        list of genes str
                cv : int
                        number of folds in cross validation, if set to 0, 1, or None cv is not done
                early_stopping_rounds : int
                        set the number of rounds for early stopping funciton of xgboost
                build_single_models : boolean or in <1,2>
                        if set to True a single model is also built that can be used for prediction 
                        and SHAP analysis
                        using 1 or 2 as inputs define if a test set is used
                seed : int
                        passed to xgb.cv to select folds
                """
                
                if genes is None:
                        genes = self.genes
                else:
                        if len(list(set(genes) - set(self.genes))):
                                raise ValueError(f"Invalid genes chosen")

                print(f"\n{self.cell_type.upper()} - ", end="")
                for gene in genes:
                        print(gene, end=" ")
                print("\n")

                if type(cv) is int and cv>1:
                        self.cv_results = pd.DataFrame()

                if not self.data_built:
                        raise RuntimeError("Data is not built call buil_datasets before building models.")

                # Check if classification or regression
                self.classifier = None
                metrics="rmse"
                stratified = False
                if self.datasets[self.genes[0]]["y_val"].dtype == "int64":
                        if self.datasets[self.genes[0]]["y_val"].max()>1:
                                metrics = "mlogloss"
                                self.classifier = "categorical"
                                stratified = True
                                model_params["objective"] = "multi:softmax"
                                model_params["num_class"] = 3
                                eval_metric = ["mlogloss"]
                        else:
                                metrics = "auc"
                                self.classifier = "binary" 
                                stratified = True 
                                model_params["objective"] = "binary:logistic" 
                                model_params["num_class"] = 1   
                                eval_metric = ["auc"]        
                else:
                        if type(model_params) == dict:
                                if "num_class" in model_params:
                                        raise ValueError("XGBoost for regression does not accept 'num_class' keyword argument.")



                for i, gene in enumerate(genes):
                        print(f"   {gene}:")

                        if type(model_params) == list:
                                params = model_params[i]
                        else:
                                params = model_params
                        
                        # Use 80% of data for cross validation
                        X = pd.concat([self.datasets[gene]["X_test"],
                                       self.datasets[gene]["X_train"]])
                        
                        Y = pd.concat([self.datasets[gene]["y_test"],
                                       self.datasets[gene]["y_train"]])

                        # Perform cross validation
                        if type(cv) is int and cv>1:
                                print(f"      Performing {cv} fold cross-validation on {gene} -", end="  ")
                                datamatrix = xgb.DMatrix(X, label=Y)

                                local_cv_result = xgb.cv(params,
                                                         datamatrix,
                                                         num_boost_round=params["n_estimators"],
                                                         early_stopping_rounds=early_stopping_rounds,  
                                                         nfold=cv, metrics=[metrics], 
                                                         stratified=stratified,
                                                         seed=seed)

                                row_name = self.cell_type + "-" + gene
                                self.cv_results = self.cv_results.append(local_cv_result.iloc[-1])
                                
                                self.cv_results = self.cv_results.rename(index={local_cv_result.shape[0]-1 : row_name})
                                print(f"{local_cv_result.shape[0]-1} iters")
                                print(f"      {metrics}: {local_cv_result.iloc[-1,2]}")


                        if build_single_models:
                                #TODO: add if in case of tuning. 
                                if type(cv) is int and cv>1:
                                        params["n_estimators"] = local_cv_result.shape[0]-1
                                        early_stopping_rounds = None

                                        update_str = f"model for {gene}, n_estimators={local_cv_result.shape[0]-1} ..."
                                else:
                                        update_str = f"model for {gene} using early stopping ..."

                                if build_single_models == 2:
                                        X_train = self.datasets[gene]["X_train"]
                                        X_test = self.datasets[gene]["X_test"]
                                        y_train = self.datasets[gene]["y_train"]
                                        y_test = self.datasets[gene]["y_test"]
                                        eval_set = [(X_train, y_train),(X_test, y_test)]
                                        self.full_training_set=False
                                
                                if build_single_models == 1:
                                        X_train = X
                                        y_train = Y
                                        eval_set = [(X, Y)]
                                        self.full_training_set=True

                                if self.classifier == "binary" or self.classifier == "categorical":
                                        single_model = xgb.XGBClassifier(**params, use_label_encoder=False)
                                        update_str = "      Building classification " + update_str
                                else:
                                        single_model = xgb.XGBRegressor(**params)
                                        eval_metric = ["rmse"]
                                        update_str = "      Building regression " + update_str
                                
                                print(update_str, end=" ")

                                single_model.fit(X_train, y_train,
                                                 eval_set=eval_set, 
                                                 early_stopping_rounds=early_stopping_rounds, 
                                                 verbose=0, eval_metric=eval_metric)
                                
                                self.models[gene] = single_model
                                print("Complete")

                print("MODEL BUILDING COMPLETE\n\n")


        def predict(self, gene, set="X_val"):
                if gene not in self.models.keys():
                        raise ValueError(f"Single model for {gene} is not built.")
                X = self.datasets[gene][set]
                return self.models[gene].predict(X)


        # TODO: Unfinished function - complete automatic tuning for each gene and cell type
        #       might not be necessary because this is very intense calculation
        def tune_model(self, start_params):
                
                for gene in self.genes:
                        pass


        def test_multiple_datasets(self, model_params, genes : list,  nan_share : list, n_inputs : list, cv=5, seed=10):
                """Test different dataset configurations
                
                genes : list
                        Which genes to test the configurations for
                nan_share : list of floats in <0,1>
                        sets upper limit for the share of NaNs for an input variable.
                n_inputs : list of ints
                        sets the number of input variables to be included.
                """

                # Test if input variables are correct
                assert type(nan_share) == list
                assert type(genes) == list
                assert type(n_inputs) == list
                for i in n_inputs:
                        assert type(i) == int
                for i in nan_share:
                        assert i < 1 and i > 0
                for gene in genes:
                        assert gene in self.genes

                combinations = [(a,b) for a in nan_share for b in n_inputs]
                
                results = []
                
                for gene in genes:
                        gene_results = pd.DataFrame()

                        self.VS.compute_corr(gene, verbose=1)

                        print(f"Fitting {cv} folds for each of {len(combinations)} data set configurations, totalling {cv*len(combinations)} fits.\n\n")
                        times = np.zeros(len(combinations))
                        for i, c in enumerate((combinations)):
                                start = time.time()
                                print(f"         Config {i+1}/{len(combinations)}")
                                print(f"         Nan share: {c[0]}, n_inputs: {c[1]} -", end="  ")
                                X, y, alpha = self.VS.extract_data(gene, nan_limit = c[0], select_n=c[1])

                                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

                                datamatrix = xgb.DMatrix(X_train, label=y_train)

                                cv_res = xgb.cv(    model_params,
                                                datamatrix,
                                                num_boost_round=500,
                                                early_stopping_rounds=20,  
                                                nfold=cv, metrics=["rmse"], 
                                                seed=seed)

                                gene_results.append(cv_res.iloc[-1])

                                gene_results = gene_results.append(cv_res.iloc[-1])
                                row_name = f"NaN: {c[0]}, N: {c[1]}, a: {alpha}"
                                gene_results = gene_results.rename(index={cv_res.shape[0]-1 : row_name})
                                end = time.time()
                                print(f"{cv_res.shape[0]-1} iters")
                                print(f"         rmse: {cv_res.iloc[-1,2]}")
                                times[i] = end-start
                                print(f"         model time : {times[i]:.0f} s")
                                print(f"         elapsed time : {times.sum():.0f} s", end="")
                                try:    
                                        eta = times[:i+1].mean()*(len(combinations)-(i+1))
                                        print(f", eta: {eta/60:.2f} min\n\n")
                                except:
                                        pass
                                
                                
                        results.append(gene_results)

                        

                if len(results)==1:
                        return results[0]
                else:
                        return results


        def build_datasets(self, diff_lim=0.5, nan_limit=0.2, select_n=1000, seed=10, verbose=1, positive_exp=False):
                self.datasets = {}

                for gene in self.genes:
                        print(f"Getting data for {gene}")
                        self.VS.compute_corr(gene, verbose=1, positive_exp=positive_exp, diff_lim=diff_lim)

                        X, y, alpha = self.VS.extract_data(gene, diff_lim=diff_lim, nan_limit=nan_limit, select_n=select_n, save_corr=False, verbose=verbose, positive_exp=positive_exp)

                        X_train, X_test, X_val, y_train, y_test, y_val = modules.train_test_val_split(X, y, random_state=seed)

                        self.datasets[gene] = {"X_train" : X_train,
                                                "X_test" : X_test,
                                                "X_val" : X_val,
                                                "y_train" : y_train,
                                                "y_test" : y_test,
                                                "y_val" : y_val,
                                                "alpha" : alpha}
                        self.data_built = True


        def get_data(self, gene):
                X = pd.concat([self.datasets[gene]["X_val"],
                                self.datasets[gene]["X_train"],
                                self.datasets[gene]["X_test"]])
                        
                Y = pd.concat([self.datasets[gene]["y_val"],
                                self.datasets[gene]["y_train"],
                                self.datasets[gene]["y_test"]])
                return X, Y

        def set_data(self, X, Y, gene, seed=10, alpha=0):
                
                X_train, X_test, X_val, y_train, y_test, y_val = modules.train_test_val_split(X, Y, random_state=seed)
                
                self.datasets[gene] = {"X_train" : X_train,
                                                "X_test" : X_test,
                                                "X_val" : X_val,
                                                "y_train" : y_train,
                                                "y_test" : y_test,
                                                "y_val" : y_val,
                                                "alpha" : alpha}
                self.data_built = True


        def plot_results(self, gene, savefig=None, figsize=(10,6), dpi=200):
                """Plot prediction results for a gene on training, testing, and validation
                
                Parameters: 
                -----------
                gene : str
                        The gene to plot results for
                
                return : fig
                """
                # Check if models are built
                if gene in self.models:

                        # Make models predict
                        model = self.models[gene]
                        
                        data = self.datasets[gene]
                        
                        y_pred_val = model.predict(data["X_val"])

                        if self.full_training_set:
                                X_train = pd.concat([self.datasets[gene]["X_val"],
                                                self.datasets[gene]["X_train"]])
                        
                                Y_train = pd.concat([self.datasets[gene]["y_val"],
                                                self.datasets[gene]["y_train"]])
                                y_pred_train = model.predict(X_train)
                        else:
                                y_pred_train = model.predict(data["X_train"])
                                Y_train = data["y_train"]
                                y_pred_test = model.predict(data["X_test"])
                        
                        # Plot predictions
                        if self.classifier is None:
                                fig, axes = plt.subplots(2,2, figsize=figsize, dpi=dpi)
                                modules.plot_prediction(Y_train, y_pred_train, fig=fig, ax=axes[0][0], title="Training predictions")
                                if not self.full_training_set:
                                        modules.plot_prediction(data["y_test"], y_pred_test, fig=fig, ax=axes[1][0], title="Test predictions")
                                else:
                                        modules.plot_prediction_alt(data["y_val"], y_pred_val, fig=fig, ax=axes[1][0], title="Validation predictions")
                                modules.plot_prediction(data["y_val"], y_pred_val, fig=fig, ax=axes[1][1], title="Validation predictions")
                                modules.plot_training_xgb(model.evals_result_, fig=fig, ax=axes[0][1], title="Training error")
                                title = "Results for " + gene + " on " + self.cell_type
                                fig.suptitle(title)
                                fig.tight_layout()
                        else:
                                fig, axes = plt.subplots(1,3,figsize=figsize, dpi=dpi)
                                if self.classifier == "binary":
                                        labels = ["DownRegulated", "UpRegulated"]
                                elif self.classifier == "categorical":
                                        labels = ["DownRegulated", "Undifferentiated", "UpRegulated"]
                                
                                
                                cf_norm = confusion_matrix(data["y_val"], y_pred_val, normalize='true')


                                modules.plot_confusion_matrix(cf_norm, labels=labels, fig=fig, ax=axes[0], sum_stats=False, title="Normalized confucion matrix")
                                cf = confusion_matrix(data["y_val"], y_pred_val)
                                modules.plot_confusion_matrix(cf, labels=labels, fig=fig, ax=axes[1])

                                modules.plot_training_xgb(model.evals_result_, fig=fig, ax=axes[2], title="Training error")
                                fig.tight_layout()
                                
                        if type(savefig) == str:
                                plt.savefig(savefig)
                                plt.close(fig)
                else:
                        raise IndexError("No models for this gene")


        def plot_validation_all_genes(self, savefig=None, figsize=(10,6), dpi=200):
                """Plot the prediction on the validation set on all the genes
                
                Parameters:
                -----------
                savefig : str
                        If set to a filepath with filename at the end the plot will be saved
                        e.g. /plots/prediction.png
                figsize : tuple of ints, (int, int)
                        set the size of the figure , length x height
                dpi : int
                        the resolution of the figure
                """
                
                if len(self.genes) == 0:
                        print("Models have not been built - use build_models with build_single_models=True")
                        return None
                elif len(self.genes) == 1:
                        raise NotImplementedError("This function does not yet work with only one gene, use plot_results insted.")
                if len(self.genes)%2:
                        n_genes = len(self.genes)+1
                else:
                        n_genes = len(self.genes)
                shape_ar = np.zeros(n_genes)

                fig, axes = plt.subplots(2,2, figsize=figsize, dpi=dpi)
                axes_flat = axes.flatten()
                for i, gene in enumerate(self.genes):
                        model = self.models[gene]
                        data = self.datasets[gene]
                        y_pred_val = model.predict(data["X_val"])
                        modules.plot_prediction(data["y_val"], y_pred_val, fig=fig, ax=axes_flat[i], title=f"{gene}: Validation Predictions")
                fig.suptitle(f"{self.cell_type} - Validation predictions for all genes")
                fig.tight_layout()

                if type(savefig) == str:
                        try: 
                                plt.savefig(savefig)
                        except:
                                raise Warning("WARNING: Invalid file path at {savefig}\n         Plot not saved.")


        def sklearn_gscv(self, model, params, cv=5, scoring="neg_root_mean_squared_error", verbose=0, gene="ESR1"):
                from sklearn.model_selection import GridSearchCV
                """
                Implement GridSearchCV for sklearn models
                
                Parameters:
                -----------
                model : sklearn estimator object
                        For example Ridge
                params : dict of lists
                        parameters to be tested in grid search"""
                

                genes = self.genes

                print(f"{self.cell_type.upper()} - ", end="")
                for gene in genes:
                        print(gene, end=" ")
                print("\n\n")

                genes = ["ESR1"]
                for gene in genes:
                        X = pd.concat([self.datasets[gene]["X_val"],
                                       self.datasets[gene]["X_train"]])
                        
                        Y = pd.concat([self.datasets[gene]["y_val"],
                                       self.datasets[gene]["y_train"]])

                        is_NaN_x = X.isnull()
                        row_has_NaN_x = is_NaN_x.any(axis=1)
                        
                        X = X[~row_has_NaN_x]
                        Y = Y[~Y.iloc[row_has_NaN_x.index]]

                        self.GSCV = GridSearchCV(model, params, scoring=scoring, n_jobs=80)
                        self.GSCV.fit(X, Y)


        def shap_analysis(self, genes=None, interaction=True):
                """
                Perform SHAP analysis for all models
                
                Parameters:
                -----------
                gene : str or list of str
                        specifiy which genes to do SHAP analysis for, if not specified
                        SHAP is done for all genes that there are prediction models for.
                """

                # Check for validity of genes
                if genes is None:
                        genes = self.genes
                elif type(genes) is list:
                        for gene in genes:
                                if gene not in self.models:
                                        raise ValueError(f"You have not built a model for this gene - {gene}")
                elif type(genes) is str:
                        if genes not in self.models:
                                        raise ValueError(f"You have not built a model for this gene - {genes}")
                        genes = [genes]
                else: 
                        raise TypeError(f"You provided {type(genes)}, but genes must be str or list of str")

                self.shapley = {}

                for gene in genes:
                        self.shapley[gene] = {}
                        explainer = shap.TreeExplainer(self.models[gene])
                        data = self.datasets[gene]
                        print(f"Calculating shap_values for {gene}...", end=" ")
                        print("Completed")
                        shap_vals = explainer.shap_values(data["X_val"])
                        self.shapley[gene]["shap_vals"] = shap_vals
                        if interaction:
                                print(f"Calculating interactions for {gene}...", end = "")
                                shap_interaction = explainer.shap_interaction_values(data["X_val"])
                                print("Completed")
                                self.shapley[gene]["shap_interaction"] = shap_interaction
                        self.shapley[gene]["explainer"] = explainer
                        
                print("Shap analysis completed")
                self.shap_analysis_complete = True


        def shap_summary(self, gene, **kwargs):
                if self.shap_analysis_complete:
                        shap.summary_plot(self.shapley[gene]["shap_vals"], self.datasets[gene]["X_val"], **kwargs)
                else:
                        print("shap_analysis must be called before plotting shap plots")


        def shap_bar(self, gene, plot_rest=False):
                if self.shap_analysis_complete:

                        modules.shap_bar(self.shapley[gene]["shap_vals"], self.datasets[gene]["X_val"], others=plot_rest)      
                else:
                        print("shap_analysis must be called before plotting shap plots")


        def shap_interaction(self, gene, max_display=10, **kwargs):
                if self.shap_analysis_complete:
                        shap.summary_plot(self.shapley[gene]["shap_interaction"], self.datasets[gene]["X_val"], max_display=max_display, plot_type="compact_dot", **kwargs)
                else:
                        print("shap_analysis must be called before plotting shap plots")


        def shap_interaction_grid(self, gene, **kwargs):
                if self.shap_analysis_complete:
                        data = self.datasets[gene]
                        shap.summary_plot(self.shapley[gene]["shap_interaction"], data["X_val"], plot_type="dot", **kwargs)
                else:
                        print("shap_analysis must be called before plotting shap plots")


        def shap_dependence(self, gene : str, genes_to_plot : list, plot_against=[]):

                if self.shap_analysis_complete:
                        if len(genes_to_plot) > 3:
                                print("\nWARNING: The plot looks nicest when plotting 3 genes\n") 

                        # Retrieve all shap values for the gene
                        shap_values = self.shapley[gene]["shap_vals"]
                        
                        data = self.datasets[gene]

                        fig, axes = plt.subplots(1,len(genes_to_plot), figsize=(20,8), sharey=True)
                        
                        # Find indices of the genes
                        for i, gene in enumerate(genes_to_plot):
                                genes_to_plot[i] = data["X_val"].columns.get_loc(gene)

                        for i, gene in enumerate(genes_to_plot):
                                shap.dependence_plot(gene,shap_values, data["X_val"], ax=axes[i], show=False)
                                pass

                        fig.tight_layout()
                else:
                        print("shap_analysis must be called before plotting shap plots")



        def shap_beeswarm(self, gene):
                if self.shap_analysis_complete:
                        shap.plots.beeswarm(self.shap_vals[gene])
                else:
                        print("shap_analysis must be called before plotting shap plots")


        @property
        def _cv_results(self):
                return self.cv_results