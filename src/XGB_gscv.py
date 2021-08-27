import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import pandas as pd

from data_handling import transpose_set_ind, correlation_reduction, root_mse


# Data handling
df_c = pd.read_pickle("/data/severs/rna_scaled_cancer.pkl")

df = transpose_set_ind(df_c)

df_reduced = correlation_reduction(df, "ESR1", limit=0.1)

X = df_reduced.drop(["ESR1"], axis=1)
y = df_reduced["ESR1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


xgb_model = xgb.XGBRegressor(objective="reg:squarederror", seed=42)

params = { 
    'eta': np.linspace(0.01, 0.5, 7),
    'objective': ['reg:squarederror'],
    'max_depth': np.arange(3,10)
}

GsCV_reg = GridSearchCV(xgb_model, params, verbose=3, scoring="neg_mean_squared_error", cv=5, n_jobs=80)

GsCV_reg.fit(X_train, y_train)

import pickle



with open("XGBoost_GSCV_result.pkl", "wb") as f:
    pickle.dump(GsCV_reg.cv_results_, f)

print("BEST MODEL:")
print(f"     Best params: {GsCV_reg.best_estimator_}")
print(f"      Best score: {GsCV_reg.best_score_}")