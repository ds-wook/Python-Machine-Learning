# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def true_fuc(X):
    return np.cos(1.5 * np.pi * X)

# %%
np.random.seed(0)
n_samples = 30

X = np.sort(np.random.rand(n_samples))
y = true_fuc(X) + np.random.rand(n_samples) * 0.1

# %%
plt.scatter(X, y)
plt.show()

# %%
plt.figure(figsize = (14, 5))
degrees = [1, 4, 15]

for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks = (), yticks = ())

    polynomial_feature = PolynomialFeatures(degree = degrees[i], include_bias=False)
    linear_regression = LinearRegression()

    pipeline = Pipeline([("polynomial_features", polynomial_feature), ("linear_regression", linear_regression)])
    pipeline.fit(X.reshape(-1, 1), y)

    # 교차 검증으로 다항 회귀를 평가
    scores = cross_val_score(pipeline, X.reshape(-1, 1), y, scoring = "neg_mean_squared_error", cv = 10)
    coefficients = pipeline.named_steps["linear_regression"].coef_
    print("Degree {0} 회귀계수는 {1} 입니다.".format(degrees[i], np.round(coefficients, 2)))
    print("Degree {0} MSE 는 {1:.2f} 입니다.".format(degrees[i], -1 * np.mean(scores)))

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label = "Model")
    plt.plot(X_test, true_fuc(X_test), "--", label = "True function")
    plt.scatter(X, y, edgecolor = "b", s = 20, label = "Samples")

    plt.xlabel("x");plt.ylabel("y");plt.xlim((0, 1));plt.ylim((-2, 2));plt.legend(loc = "best")
    plt.title("Degree{}\n MSE = {:.2e}(+ / - {:.2e})".format(degrees[i], -scores.mean(), scores.std()))
plt.show()

# %%
