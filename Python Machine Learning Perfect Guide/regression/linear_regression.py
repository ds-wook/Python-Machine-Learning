# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston

# %%
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)

bostonDF["PRICE"] = boston.target
print("Boston dataset size : ", bostonDF.shape)
bostonDF.head()

# %%
fig, axs = plt.subplots(figsize = (16, 8), ncols = 4, nrows = 2)
lm_feature = ["RM", "ZN", "INDUS", "NOX", "AGE", "PTRATIO", "LSTAT", "RAD"]

for i, feature in enumerate(lm_feature):
    row = i // 4
    col = i % 4
    sns.regplot(x = feature, y = "PRICE", data = bostonDF, ax = axs[row][col])

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target = bostonDF["PRICE"]
X_data = bostonDF.drop("PRICE", axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.3, random_state = 156)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)

print("MSE : {0:.3f}, RMSE : {1:.3f}".format(mse, rmse))
print("Variance score : {0:.3f}".format(r2_score(y_test, y_preds)))

# %%
print("절편 값 : ", lr.intercept_)
print("회귀 계수값 : ", np.round(lr.coef_, 1))

# %%
coeff = pd.Series(data = np.round(lr.coef_, 1), index = X_data.columns)
coeff.sort_values(ascending = False)

# %%
from sklearn.model_selection import cross_val_score

y_target = bostonDF["PRICE"]
X_data = bostonDF.drop("PRICE", axis = 1, inplace = False)
lr = LinearRegression()

neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring = "neg_mean_squared_error", cv = 5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print("5 fold's Negative MSE scores : ", np.round(neg_mse_scores, 2))
print("5 fold's RMSE scores : ", np.round(rmse_scores, 2))
print("5 fold's mean RMSE : {0:.3f}".format(avg_rmse))

# %%
