# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes = True)
x = np.random.normal(0, 1, size = 30)
print(x)
sns.distplot(x)
plt.show()
# %%
sns.distplot(x, rug = True)
plt.show()

# %%
sns.distplot(x, kde = False, rug = True)
plt.show()

# %%
sns.distplot(x, hist = False, rug = True)
plt.show()

# %%
from scipy import stats

bandwidth = 1.06 * x.std() * x.size ** (-1 / 5)
support = np.linspace(-4, 4, 200)

kernels = []

for x_i in x:
    kernel = stats.norm(x_i, bandwidth).pdf(support)
    kernels.append(kernel)
    plt.plot(support, kernel, color = "r")

sns.rugplot(x, color = ".2", linewidth = 3)
plt.show()

# %%
from scipy.integrate import trapz

density = np.sum(kernels, axis = 0)
density /= trapz(density, support)
plt.plot(support, density)

# %%
sns.kdeplot(x, shade = True)
plt.show()

# %%
sns.kdeplot(x)
sns.kdeplot(x, bw = 0.2, label = "bw:0.2")
sns.kdeplot(x, bw = 2, label = "bw : 2")
plt.legend()

# %%
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift

X, y = make_blobs(n_samples=200, n_features=2, centers = 3, cluster_std=0.8, random_state=0)
meanshift = MeanShift(bandwidth=0.9)
cluster_labels = meanshift.fit_predict(X)
print("cluster labels 유형 : ", np.unique(cluster_labels))

# %%
meanshift = MeanShift(bandwidth=1)
cluster_labels = meanshift.fit_predict(X)
print("cluster labels 유형", np.unique(cluster_labels))

# %%
from sklearn.cluster import estimate_bandwidth

bandwidth = estimate_bandwidth(X, quantile = 0.25)
print("bandwidth 값 : ", np.round(bandwidth, 2))

# %%
import pandas as pd

clusterDF = pd.DataFrame(data = X, columns = ["ftr1", "ftr2"])
clusterDF["target"] = y

best_bandwidth = estimate_bandwidth(X, quantile = 0.25)

meanshift = MeanShift(best_bandwidth)
cluster_labels = meanshift.fit_predict(X)
print("cluster labels 유형 : ", np.unique(cluster_labels))

# %%
clusterDF["meanshift_label"] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ["o", "s", "^", "x", "*"]

for label in unique_labels:
    label_cluster = clusterDF[clusterDF["meanshift_label"] == label]
    center_x_y = centers[label]
    
    plt.scatter(x = label_cluster["ftr1"], y = label_cluster["ftr2"], edgecolor = "k", marker=markers[label])
    plt.scatter(x = center_x_y[0], y = center_x_y[1], s = 200, color = "white", marker = markers[label])
    plt.scatter(x = center_x_y[0], y = center_x_y[1], s = 70, color = "k", edgecolor = "k", marker = "$%d$" % label)
plt.show()

# %%
print(clusterDF.groupby("target")["meanshift_label"].value_counts())

# %%
