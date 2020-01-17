# %%
import pandas as pd
df = pd.read_excel("./data/pca_credit_card.xls", sheet_name="Data", header = 1)
print(df.shape)
df.head(3)

# %%
df.rename(columns = {"PAY_0" : "PAY_1", "default payment next month" : "default"}, inplace = True)
y_target = df["default"]
X_features = df.drop(["ID", "default"], axis = 1)

# %%
y_target.value_counts()

# %%
X_features.info()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

corr = X_features.corr()
plt.figure(figsize = (14, 14))
sns.heatmap(corr, annot = True, fmt = ".1g")
plt.show()

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cols_bill = ["BILL_AMT" + str(i) for i in range(1, 7)]
print("대상 속성명 : ", cols_bill)

scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill])
pca = PCA(n_components = 2)
pca.fit(df_cols_scaled)

print("PCA Component별 변동성 : ", pca.explained_variance_ratio_)

# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rcf = RandomForestClassifier(n_estimators=300, random_state = 156)
scores = cross_val_score(rcf, X_features, y_target, scoring = "accuracy", cv = 3)

print("CV = 3인 경우의 개별 Fold 세트별 정확도 : ", scores)
print("평균 정확도 : {0:.4f}".format(np.mean(scores)))


# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(X_features)

pca = PCA(n_components = 6)
df_pca = pca.fit_transform(df_scaled)
scores_pca = cross_val_score(rcf, df_pca, y_target, scoring = "accuracy", cv=3)
print("CV = 3인 경우의 개별 Fold 세트별 정확도 : ", scores)
print("평균 정확도 : {0:.4f}".format(np.mean(scores)))

# %%
