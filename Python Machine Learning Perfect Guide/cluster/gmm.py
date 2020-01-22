# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

irisDF = pd.DataFrame(data = iris.data, columns = feature_names)
irisDF["target"] = iris.target

# %%
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components = 3, random_state = 0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

irisDF["gmm_cluster"] = gmm_cluster_labels

iris_result = irisDF.groupby(["target"])["gmm_cluster"].value_counts()
iris_result

# %%
iris.target_names

# %%
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter = 300, random_state = 0).fit(iris.data)
kmeans_cluster_labels = kmeans.predict(iris.data)
irisDF["kmeans_cluster"] = kmeans_cluster_labels
iris_result = irisDF.groupby(["target"])["kmeans_cluster"].value_counts()
iris_result

# %%
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
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
        
        plt.scatter(x = label_cluster["ftr1"], y = label_cluster["ftr2"], s = 70, edgecolors= "k", marker = markers[label])

        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x = center_x_y[0], y = center_x_y[1], s = 250, color = "white", alpha = 0.9, edgecolor = "k", marker = markers[label])
            plt.scatter(x = center_x_y[0], y = center_x_y[1], s = 70, color = "k", edgecolor = "k", marker = "$%d$" % label)
    
    if isNoise:
        legend_loc = "upper center"
    else:
        legend_loc = "upper right"
    
    plt.legend(loc = legend_loc)
    plt.show()

# %%
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=0)

transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)

clusterDF = pd.DataFrame(data = X_aniso, columns = ["ftr1", "ftr2"])
clusterDF["target"] = y

visualize_cluster_plot(None, clusterDF, "target", iscenter = False)

# %%
kmeans = KMeans(3, random_state = 0)
kmeans_label = kmeans.fit_predict(X_aniso)
clusterDF["kmeans_label"] = kmeans_label

visualize_cluster_plot(kmeans, clusterDF, "kmeans_label", iscenter = True)

# %%
gmm = GaussianMixture(n_components = 3, random_state = 0)
gmm_label = gmm.fit(X_aniso).predict(X_aniso)
clusterDF["gmm_label"] = gmm_label

visualize_cluster_plot(gmm, clusterDF, "gmm_label", iscenter = False)

# %%
print("### KMeans Clustering ###")
print(clusterDF.groupby("target")["kmeans_label"].value_counts())
print("\n### Gaussian Mixture Clustering ###")
print(clusterDF.groupby("target")["gmm_label"].value_counts())

# %%
