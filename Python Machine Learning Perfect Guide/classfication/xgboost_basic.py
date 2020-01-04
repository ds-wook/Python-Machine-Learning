# %%
import xgboost
print(xgboost.__version__)

# %%
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# %%
dataset = load_breast_cancer()
X_features = dataset.data
y_label = dataset.target

# %%
cancer_df = pd.DataFrame(data = X_features, columns = dataset.feature_names)
cancer_df["target"] = y_label
cancer_df.head(3)

# %%
print(dataset.target_names)
print(cancer_df["target"].value_counts())

# %%
X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size = 0.2, random_state = 156)
print(X_train.shape, X_test.shape)

# %%
dtrain = xgb.DMatrix(data = X_train, label = y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)

# %%
params = {
    "max_depth" : 3,
    "eta" : 0.1,
    "objective" : "binary:logistic",
    "eval_metric" : "logloss",
    "early_stoppings" : 100
}

num_rounds = 400
# %%
wlist = [(dtrain, "train"), (dtest, "eval")]
xgb_model = xgb.train(params = params, dtrain = dtrain, num_boost_round=num_rounds, evals = wlist)



# %%
pred_probs = xgb_model.predict(dtest)
print("predict() 수행 결괏값을 10개만 표시, 예측 확률값으로 표시됨")
print(np.round(pred_probs[:10], 3))

preds = [1 if x > 0.5 else 0 for x in pred_probs]
print("예측값 10개만 표시 : ", preds[:10])

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test, pred = None, pred_proba = None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print("오차행렬")
    print(confusion)
    print("정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 {2:.4f}, F1 : {3:.4f}, AUC : {4:.4f}".format(accuracy, precision, recall, f1, roc_auc))

# %%
get_clf_eval(y_test, preds, pred_probs)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (10, 12))
plot_importance(xgb_model, ax = ax)

# %%
from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(X_train, y_train)
w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]
# %%
get_clf_eval(y_test, w_preds, w_pred_proba)

# %%
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
evals = [(X_test, y_test)]
xgb_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", eval_set=evals, verbose=True)
ws100_preds = xgb_wrapper.predict(X_test)
ws100_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

# %%
get_clf_eval(y_test, ws100_preds, ws100_pred_proba)

# %%
fig, ax = plt.subplots(figsize = (10, 12))
plot_importance(xgb_wrapper, ax = ax)
plt.show()

# %%
