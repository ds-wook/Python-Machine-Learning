# %%
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

irisDF = pd.DataFrame(data = iris.data, columns = feature_names)
irisDF["target"] = iris.target
irisDF.head()

# %%
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps = 0.6, min_samples = 8, metric = "euclidean")
dbscan_labels = dbscan.fit_predict(iris.data)

irisDF["dbscan_cluster"] = dbscan_labels

iris_result = irisDF.groupby(["target"])["dbscan_cluster"].value_counts()
iris_result

# %%
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter = True):
    if iscenter:
        centers = clusterobj.cluster_centers_
    
    unique_labels = np.unique(dataframe[label_name].values)
    markers = ["o", "s", "^", "x", "*"]
    isNoise = False
    
    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name] == label]
        if label == -1:
            cluster_legend = "Noise"
            isNoise = True
        else:
            cluster_legend = "Cluster" + str(label)

        plt.scatter(x = label_cluster["ftr1"], y = label_cluster["ftr2"], s = 70, edgecolors="k", marker=markers[label], label = cluster_legend)

        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x = center_x_y[0], y = center_x_y[1], s = 250, color = "white", alpha = 0.9, edgecolors="k", marker = markers[label])
            plt.scatter(x = center_x_y[0], y = center_x_y[1], s = 70, color = "k", edgecolors="k", marker = "$%d$" % label)
        
    if isNoise:
        legend_loc = "upper_center"
    else:
        legend_loc = "upper right"
        
    plt.legend(loc = legend_loc)
    plt.show()

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=0)
pca_transformed = pca.fit_transform(iris.data)

irisDF["ftr1"] = pca_transformed[:, 0]
irisDF["ftr2"] = pca_transformed[:, 1]

visualize_cluster_plot(dbscan, irisDF, "dbscan_cluster", iscenter = False)

# %%
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps = 0.8, min_samples=8, metric="euclidean")
dbscan_labels = dbscan.fit_predict(iris.data)

irisDF["dbscan_cluster"] = dbscan_labels
irisDF["target"] = iris.target

iris_result = irisDF.groupby(["target"])["dbscan_cluster"].value_counts()

print(iris_result)
visualize_cluster_plot(dbscan, irisDF, "dbscan_cluster", iscenter = False)

# %%
from sklearn.datasets import make_circles

X, y = make_circles(n_samples = 1000, shuffle = True, noise = 0.05, random_state = 0, factor = 0.5)
clusterDF = pd.DataFrame(data = X, columns = ["ftr1", "ftr2"])
clusterDF["target"] = y
visualize_cluster_plot(None, clusterDF, "target", iscenter = False)

# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, max_iter = 1000, random_state = 0)
kmeans_labels = kmeans.fit_predict(X)
clusterDF["kmeans_cluster"] = kmeans_labels
visualize_cluster_plot(kmeans, clusterDF, "kmeans_cluster", iscenter = True)

# %%
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components = 2, random_state = 0)
gmm_label = gmm.fit(X).predict(X)
clusterDF["gmm_cluster"] = gmm_label

visualize_cluster_plot(gmm, clusterDF, "gmm_cluster", iscenter = False)

# %%
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps = 0.2, min_samples = 10, metric = "euclidean")
dbscan_labels = dbscan.fit_predict(X)
clusterDF["dbscan_cluster"] = dbscan_labels
visualize_cluster_plot(dbscan, clusterDF, "dbscan_cluster", iscenter = False)

# %%
