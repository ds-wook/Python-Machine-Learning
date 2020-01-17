# %%
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
irisDF = pd.DataFrame(iris.data, columns = columns)
irisDF["target"] = iris.target
irisDF.head(3)

# %%
makers = ["^", "s", "o"]

for i, marker in enumerate(makers):
    x_axis_data = irisDF[irisDF["target"] == i]["sepal_length"]
    y_axis_data = irisDF[irisDF["target"] == i]["sepal_width"]
    plt.scatter(x_axis_data, y_axis_data, marker = marker, label = iris.target_names[i])

plt.legend()
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.show()
# %%
from sklearn.preprocessing import StandardScaler
iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:, :-1])

# %%
iris_scaled.shape

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)
print(iris_pca.shape)

# %%
pca_columns = ["pca_component_1", "pca_component_2"]
irisDF_pca = pd.DataFrame(iris_pca, columns = pca_columns)
irisDF_pca["target"] = iris.target
irisDF_pca.head(3)

# %%
markers = ["^", "s", "o"]

for i, marker in enumerate(markers):
    x_axis_data = irisDF_pca[irisDF_pca["target"] == i]["pca_component_1"]
    y_axis_data = irisDF_pca[irisDF_pca["target"] == i]["pca_component_2"]
    plt.scatter(x_axis_data, y_axis_data, marker = marker, label = iris.target_names[i])
plt.legend()
plt.xlabel("pca_component_1")
plt.ylabel("pca_component_2")
plt.show()

# %%
print(pca.explained_variance_ratio_)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rcf = RandomForestClassifier(random_state=156)
pca_X = irisDF_pca[["pca_component_1", "pca_component_2"]]
scores_pca = cross_val_score(rcf, pca_X, iris.target, scoring = "accuracy", cv = 3)
print(np.mean(scores_pca))

# %%
