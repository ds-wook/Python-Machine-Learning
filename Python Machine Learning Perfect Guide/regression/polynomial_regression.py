# %%
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.arange(4).reshape(2, 2)
print("일차 단항식 계수 feature : \n", X)

poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr = poly.transform(X)
print("변환된 2차 다항식 계수 feature : \n", poly_ftr)

# %%
def polynomial_func(X):
    y = 1 + 2 * X[:, 0] + 3 * X[:, 0] ** 2 + 4 * X[:, 1] ** 3
    return y

# %%
X = np.arange(4).reshape(2, 2)
print("일차 단항식 계수 feature : \n", X)
y = polynomial_func(X)
print("삼차 다항식 결정값 : \n", y)

poly_ftr = PolynomialFeatures(degree=3).fit_transform(X)
print("삼차 다항식 계수 feature : \n", poly_ftr)


# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(poly_ftr, y)
print("Polynomial 회귀 계수\n", np.round(model.coef_, 2))
print("Polynomial 회귀 Shape : ", model.coef_.shape)

# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

def polynomial_func(X):
    y = 1 + 2 * X[:, 0] + 3 * X[:, 0] ** 2 + 4 * X[:, 1] ** 3
    return y

model = Pipeline([("poly", PolynomialFeatures(degree=3)), ("linear", LinearRegression())])

X = np.arange(4).reshape(2, 2)
y = polynomial_func(X)

model = model.fit(X, y)
print("Polynomial 회귀 계수 \n", np.round(model.named_steps["linear"].coef_, 2))

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# %%
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)

bostonDF["PRICE"] = boston.target
print("Boston dataset size : ", bostonDF.shape)

y_target = bostonDF["PRICE"]
X_data = bostonDF.drop("PRICE", axis = 1, inplace = False)

X_train,  X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.2, random_state = 156)

p_model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias = False)), ("linear", LinearRegression())])
p_model.fit(X_train, y_train)
y_preds = p_model.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)

print("MSE : {0:.3f}, RMSE : {1:.3f}".format(mse, rmse))
print("Variance score : {0:.3f}".format(r2_score(y_test, y_preds)))
# %%
X_train_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train, y_train)
X_train_poly.shape, X_train.shape

# %%
