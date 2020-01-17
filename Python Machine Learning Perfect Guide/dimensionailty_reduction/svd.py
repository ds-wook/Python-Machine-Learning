# %%
import numpy as np
from numpy.linalg import svd

np.random.seed(121)
a = np.random.randn(4, 4)
print(np.round(a, 3))

# %%
U, Sigma, Vt = svd(a)
print(U.shape, Sigma.shape, Vt.shape)
print("U matrix : \n", np.round(U, 3))
print("Sigma Value : \n", np.round(Sigma, 3))
print("V transpose matrix : \n", np.round(Vt, 3))

# %%
Sigma_mat = np.diag(Sigma)
print(Sigma_mat)
a_ = np.dot(np.dot(U, Sigma_mat), Vt)
print(np.round(a_, 3))

# %%
a[2] = a[0] + a[1]
a[3] = a[0]
print(np.round(a, 3))

# %%
U, Sigma, Vt = svd(a)
print(U.shape, Sigma.shape, Vt.shape)
print("Sigma value : \n", np.round(Sigma, 3))

# %%
U_ = U[:, :2]
Sigma_ = np.diag(Sigma[:2])
Vt_ = Vt[:2]
print(U_.shape, Sigma_.shape, Vt_.shape)
a_ = np.dot(np.dot(U_, Sigma_), Vt_)
print(np.round(a_, 3))

# %%
from scipy.sparse.linalg import svds
from scipy.linalg import svd

np.random.seed(121)
matrix = np.random.random((6, 6))
print("원본 행렬 : \n", matrix)
U, Sigma, Vt = svd(matrix, full_matrices=False)
print("\n 분해 행렬 차원 : ", U.shape, Sigma.shape, Vt.shape)
print("\n Sigma값 행렬 : ", Sigma)

num_components = 4
U_tr, Sigma_tr, Vt_tr = svds(matrix, k = num_components)
print("\n Truncated SVD 분해 행렬 차원 : ", U_tr.shape, Sigma_tr.shape, Vt_tr.shape)
print("\n Truncated SVD Sigma값 행렬 : ", Sigma_tr)
matrix_tr = np.dot(np.dot(U_tr, np.diag(Sigma_tr)), Vt_tr)
print("\n Trucated SVD로 분해 후 복원 행렬 : \n", matrix_tr)

# %%
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_ftrs = iris.data


tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_ftrs)
iris_tsvd = tsvd.transform(iris_ftrs)

plt.scatter(x = iris_tsvd[:, 0], y = iris_tsvd[:, 1], c = iris.target)
plt.xlabel("TruncatedSVD Component 1")
plt.ylabel("TruncatedSVD Component 2")
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_ftrs)

tsvd = TruncatedSVD(n_components = 2)
tsvd.fit(iris_scaled)
iris_tsvd = tsvd.transform(iris_scaled)

pca = PCA(n_components = 2)
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)

fig, (ax1, ax2) = plt.subplots(figsize = (9, 4), ncols = 2)
ax1.scatter(x = iris_tsvd[:, 0], y = iris_tsvd[:, 1], c = iris.target)
ax2.scatter(x = iris_pca[:,0], y = iris_pca[:, 1], c = iris.target)
ax1.set_title("Truncated SVD Transformed")
ax2.set_title("PCA Transformed")
plt.show()

# %%
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_ftrs = iris.data
nmf = NMF(n_components = 2)

nmf.fit(iris_ftrs)
iris_nmf = nmf.transform(iris_ftrs)

plt.scatter(x = iris_nmf[:, 0], y = iris_nmf[:, 1], c = iris.target)
plt.xlabel("NMF Component 1")
plt.ylabel("NMF Component 2")
plt.show()
# %%
