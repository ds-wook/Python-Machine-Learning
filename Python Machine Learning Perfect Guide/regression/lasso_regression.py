# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, ElasticNet, Ridge
# %%
boston = load_boston()

bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)

bostonDF["PRICE"] = boston.target
print("boston dataset size : ", bostonDF.shape)

y_target = bostonDF["PRICE"]
X_data = bostonDF.drop("PRICE", axis = 1, inplace = False)
# %%
def get_linear_reg_eval(model_name, params = None, X_data_n = None, y_target_n = None, verbose = True):
    coeff_df = pd.DataFrame()
    if verbose : print("###### {} ######".format(model_name))
    for param in params:
        if model_name == "Ridge" : model = Ridge(alpha=param)
        elif model_name == "Lasso" : model = Lasso(alpha = param)
        elif model_name == "ElasticNet" : model = ElasticNet(alpha = param, l1_ratio = 0.7)
        neg_mse_scores = cross_val_score(model, X_data_n, y_target_n, scoring="neg_mean_squared_error", cv = 5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))

        print("alpha {0} 일 때 5 폴드 세트의 평균 RMSE : {1:.3f}".format(param, avg_rmse))
        model.fit(X_data, y_target)

        coeff = pd.Series(data = model.coef_, index = X_data.columns)
        colname = "alpha:" + str(param)
        coeff_df[colname] = coeff
    return coeff_df

# %%
lasso_alphas = [0.07, 0.1, 0.5, 1, 3]
coeff_lasso_df = get_linear_reg_eval("Lasso", params = lasso_alphas, X_data_n = X_data, y_target_n = y_target)

# %%
sort_column = "alpha:" + str(lasso_alphas[0])
coeff_lasso_df.sort_values(by = sort_column, ascending = False)
# %%
elastic_alphas = [0.07, 0.1, 0.5, 1, 3]
coeff_elastic_df = get_linear_reg_eval("ElasticNet", params = elastic_alphas, X_data_n = X_data, y_target_n = y_target)

# %%
sort_column = "alpha:" + str(elastic_alphas[0])
coeff_elastic_df.sort_values(by = sort_column, ascending = False)

# %%
