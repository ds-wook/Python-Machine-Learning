#%%
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()

irisDF = pd.DataFrame(data = iris.data, columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"])
irisDF.head(3)

# %%
kmeans = KMeans(n_clusters = 3, init = "k-means++", max_iter = 300, random_state = 0)
kmeans.fit(irisDF)

# %%
print(kmeans.labels_)

# %%
irisDF["cluster"] = kmeans.labels_

# %%
irisDF["target"] = iris.target
iris_result = irisDF.groupby(["target", "cluster"])["sepal_length"].count()
print(iris_result)

# %%
iris.target_names

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)

irisDF["pca_x"] = pca_transformed[:, 0]
irisDF["pca_y"] = pca_transformed[:, 1]
irisDF.head(3) 

# %%
plt.scatter(x = irisDF.loc[:, "pca_x"], y = irisDF.loc[:, "pca_y"], c = irisDF["cluster"])
plt.show()

# %%
marker0_ind = irisDF[irisDF["cluster"] == 0].index
marker1_ind = irisDF[irisDF["cluster"] == 1].index
marker2_ind = irisDF[irisDF["cluster"] == 2].index

plt.scatter(x = irisDF.loc[marker0_ind, "pca_x"], y = irisDF.loc[marker0_ind, "pca_y"], marker = "o")
plt.scatter(x = irisDF.loc[marker1_ind, "pca_x"], y = irisDF.loc[marker1_ind, "pca_y"], marker = "s")
plt.scatter(x = irisDF.loc[marker2_ind, "pca_x"], y = irisDF.loc[marker2_ind, "pca_y"], marker = "^")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("3 Clusters Visualization by 2 PCA Components")
plt.show()

# %%
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
print(X.shape, y.shape)

unique, counts = np.unique(y, return_counts=True)
print(unique, counts)

# %%
clusterDF = pd.DataFrame(data = X, columns = ["ftr1", "ftr2"])
clusterDF["target"] = y
clusterDF.head(3)

# %%
target_list = np.unique(y).tolist()
markers = ["o", "s", "^", "P", "D", "H", "x"]

for target in target_list:
    target_cluster = clusterDF[clusterDF["target"] == target]
    plt.scatter(x = target_cluster["ftr1"], y = target_cluster["ftr2"], edgecolors="k", marker = markers[target])
plt.show()

# %%
plt.scatter(x = clusterDF["ftr1"], y = clusterDF["ftr2"], edgecolors="k", c=y)
plt.show()

# %%
kmeans = KMeans(n_clusters=3, init = "k-means++", max_iter = 200, random_state=0)
cluster_labels = kmeans.fit_predict(X)
clusterDF["kmeans_label"] = cluster_labels
centers = kmeans.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ["o", "s", "^", "P", "D", "H", "x"]

for label in unique_labels:
    label_cluster = clusterDF[clusterDF["kmeans_label"] == label]
    center_x_y = centers[label]
    plt.scatter(x=label_cluster["ftr1"], y = label_cluster["ftr2"], edgecolor = "k", marker = markers[label])

    plt.scatter(x = center_x_y[0], y = center_x_y[1], s = 200, color = "white", alpha = 0.9, edgecolor = "k", marker=markers[label])
    plt.scatter(x = center_x_y[0], y  = center_x_y[1], s = 70, color = "k", edgecolor = "k", marker = "$%d$" % label)
plt.show()
# %%
print(clusterDF.groupby("target")["kmeans_label"].value_counts())

# %%
