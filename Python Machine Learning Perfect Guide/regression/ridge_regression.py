# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# %%
boston = load_boston()

bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)

bostonDF["PRICE"] = boston.target
print("boston dataset size : ", bostonDF.shape)

y_target = bostonDF["PRICE"]
X_data = bostonDF.drop("PRICE", axis = 1, inplace = False)

# %%
ridge = Ridge(alpha = 10)
neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)
print("5 folds 의 개별 Negative MSE scores : ", np.round(neg_mse_scores, 3))
print("5 folds 의 개별 RMSE scores : ", np.round(rmse_scores, 3))
print("5 folds 의 평균 RMSE : {0:.3f}".format(avg_rmse))

# %%
alphas = [0, 0.1, 1, 10, 100]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)

    neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
    avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
    print("alpha {0} 일때 5 folds의 평균 RMSE : {1:.3f}".format(alpha, avg_rmse))

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(figsize = (18, 6), nrows = 1, ncols = 5)
coeff_df = pd.DataFrame()

for pos, alpha in enumerate(alphas):
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_data, y_target)

    coeff = pd.Series(data = ridge.coef_, index = X_data.columns)
    colname = "alpha" + str(alpha)
    coeff_df[colname] = coeff

    coeff = coeff.sort_values(ascending = False)
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-3, 6)
    sns.barplot(x = coeff.values, y = coeff.index, ax = axs[pos])

plt.show()

# %%
ridge_alphas = [0, 0.1, 1, 10, 100]

sort_column = "alpha" + str(ridge_alphas[0])
coeff_df.sort_values(by=sort_column, ascending = False)

# %%
