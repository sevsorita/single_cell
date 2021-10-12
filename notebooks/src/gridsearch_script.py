from xgboost import XGBRegressor
import pandas as pd

from model_selection import XGBGridSearch
from data_handling import VariableSelector
import modules


"""
This script tunes the model
"""


if __name__ == "__main__":


    params = {
                'objective': 'reg:squarederror',
                'importance_type': 'gain',
                'learning_rate': 0.02,
                'max_depth': 4,
                'min_child_weight': 5,
                'n_estimators': 150,
                'n_jobs': 40,
                'subsample': 1,
                'verbosity': 0,
                'seed': 1,
                'silent': True
                }

    param_dist = {
                'eta' : [0.02, 0.01, 0.005],
                'subsample' : [0.7, 0.8, 0.9, 1],                
                'max_depth' : [2, 3, 4, 5],         
                'min_child_weight' : [1, 2, 3, 4],    
                }

    VS = VariableSelector('Cancer_baseline')

    gene = "ESR1"
    diff_lim = 0.5

    VS.compute_corr(gene, verbose=1, positive_exp=False, diff_lim=diff_lim)

    X, y, alpha = VS.extract_data(gene, diff_lim=diff_lim, nan_limit=0.2, select_n=1000, save_corr=False, verbose=1, positive_exp=False)

    X_train, X_test, X_val, y_train, y_test, y_val = modules.train_test_val_split(X, y, random_state=10)
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])


    tuning_model = XGBRegressor(**params)
    
    GS = XGBGridSearch(tuning_model, param_dist, n_jobs=40, cv=5)

    search1 = GS.fit(X_train, y_train)

    print("BEST_PARAMATERS:")
    for key in GS.best_params:
        if key in param_dist.keys():
            print(f"    {key}  :  {GS.best_params[key]}")
    print(f"n_estimators = {GS.iterations}")
    
    params['n_estimators'] = GS.iterations
    params['learning_rate'] = GS.best_params['learning_rate']

    

    param_dist_2 = {
                'colsample_bytree' : [0.7, 0.8],
                'subsample' : [0.7],                #[0.7, 0.8, 0.9, 1],
                'max_depth' : [4],                  #list(range(1,9, 2)),
                'min_child_weight' : [3],           #list(range(1,9, 2))
                }



