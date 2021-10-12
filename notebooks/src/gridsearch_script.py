from model_selection import XGBGridSearch
from xgboost import XGBRegressor
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
                'eta' : [0.02, 0.01, 0.04, 0.1],
                'subsample' : [0.8, 1],                
                'max_depth' : [3, 5, 7],         
                'min_child_weight' : [1, 3, 5, 7],    
                }

    VS = VariableSelector('Cancer_baseline')

    gene = "ESR1"
    diff_lim = 0.5

    VS.compute_corr(gene, verbose=1, positive_exp=False, diff_lim=diff_lim)

    X, y, alpha = VS.extract_data(gene, diff_lim=diff_lim, nan_limit=0.2, select_n=1000, save_corr=False, verbose=1, positive_exp=False)

    X_train, X_test, X_val, y_train, y_test, y_val = modules.train_test_val_split(X, y, random_state=10)

    tuning_model = XGBRegressor(**params)

    GS = XGBGridSearch(tuning_model, param_dist, n_jobs=40, cv=5)

    search1 = GS.fit(X_train, y_train)

    params['n_estimators'] = GS.iterations
    params['learning_rate'] = GS.best_params['learning_rate']

    print("BEST_PARAMATERS:")
    for key in GS.best_params:
        if key in param_dist.keys():
            print(f"    {key}  :  {GS.best_params[key]}")

    
    

    param_dist_2 = {
                'colsample_bytree' : [0.7, 0.8],
                'subsample' : [0.7],                #[0.7, 0.8, 0.9, 1],
                'max_depth' : [4],                  #list(range(1,9, 2)),
                'min_child_weight' : [3],           #list(range(1,9, 2))
                }



