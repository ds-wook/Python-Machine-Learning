# %%
import numpy as np

# %%
def n_size_ndarray_creation(n, dtype=np.int):
    X = np.arange(n ** 2, dtype = dtype).reshape(-1, n)
    return X
n_size_ndarray_creation(5, np.float64)
# %%
def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
    if type == 0:
        X = np.zeros(shape = shape, dtype = dtype)
    elif type == 1:
        X = np.ones(shape = shape, dtype=dtype)
    elif type == 99:
        X = np.empty(shape = shape, dtype = dtype)
    else:
        return False
    return X
zero_or_one_or_empty_ndarray((2, 2), 100)
# %%
def change_shape_of_ndarray(X, n_row):
    try:
        return X.flatten() if n_row == 1 else X.reshape(n_row, -1)
    except:
        return False
X = np.ones((32,32), dtype=np.int)
change_shape_of_ndarray(X, 16)
# %%
def concat_ndarray(X_1, X_2, axis):
    try:
        X_1 = X_1.reshape(1, -1) if X_1.ndim == 1 else X_1
        X_2 = X_2.reshape(1, -1) if X_2.ndim == 1 else X_2
        return np.concatenate((X_1, X_2), axis = axis)
    except:
        return False

# %%
def normalize_ndarray(X, axis=99, dtype=np.float32):
    if axis == 0:
        X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    elif axis == 1:
        # X = (X - X.mean(axis = 1).reshape(-1, 1)) / X.std(axis = 1).reshape(-1, 1)
        X = ((X.T - X.mean(axis = 1)) / X.std(axis = 1)).T
    else:
        X = (X - X.mean()) / X.std()
    return X
X = np.arange(12, dtype=np.float32).reshape(6,2)
normalize_ndarray(X, 1)
# %%
def save_ndarray(X, filename="test.npy"):
    np.save(file=filename, arr = X)
np.load("test.npy")
# %%
def boolean_index(X, condition):
    condition = eval(str("X") + condition)
    return np.where(condition)
X = np.arange(32, dtype=np.float32).reshape(4, -1)
boolean_index(X, "== 3")
# %%
def find_nearest_value(X, target_value):
    return X[np.argmin(np.abs(X - target_value))]

# %%
def get_n_largest_values(X, n):
    return X[np.argsort(X)[::-1]][:n]
