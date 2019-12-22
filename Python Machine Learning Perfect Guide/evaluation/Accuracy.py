# %% 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fillna(df):
    df["Age"].fillna(df["Age"].mean(), inplace= True)
    df["Cabin"].fillna("N", inplace = True)
    df["Embarked"].fillna("N", inplace = True)
    df["Fare"].fillna(0, inplace = True)

    return df

def drop_features(df):
    df.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)
    return df

def format_features(df):
    df["Cabin"] = df["Cabin"].str[:1]
    features = ["Cabin", "Sex", "Embarked"]

    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# %%
titanic_df = pd.read_csv("./titanic/train.csv")
y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop("Survived", axis = 1)

X_titanic_df = transform_features(X_titanic_df)

# %%
from sklearn.base import BaseEstimator

class MyDummyClassifier(BaseEstimator):
    # fit()메소드는 아무것도 학습하지 않음
    def fit(self, X, y = None):
        pass
    # predict()메서드는 단순히 Sex feature가 1이면 0 아니면 1로 예측함
    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if X["Sex"].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1
        
        return pred

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size = 0.2, random_state = 0)

# %%
myclf = MyDummyClassifier()
myclf.fit(X_train, y_train)
mypredictions = myclf.predict(X_test)
print("Dummy Classifier의 정확도는 {0:.4f}".format(accuracy_score(y_test, mypredictions)))

# %%
from sklearn.datasets import load_digits

class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass
    # 입력값으로 들어오는 X 데이터 세트의 크기만큼 모두 0값으로 만들어서 반환
    def predict(self, X):
        return np.zeros((len(X), 1), dtype = bool)

# %%
digits = load_digits()

# digits 번호가 7번이면 True 이를 astype(int)로 1 반환 아니면 False이고 0 반환
y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state = 11)

# %%
print("레이블 테스트 세트 크기 : ", y_test.shape)
print("테스트 세트 레이블 0 과 1의 분포도")
print(pd.Series(y_test).value_counts())

#Dummy Classifier로 학습/예측/정확도 평가
fakeclf = MyFakeClassifier()
fakeclf.fit(X_train, y_train)
fakepred = fakeclf.predict(X_test)

print("모든 예측을 0으로 하여도 정확도는 : {0:.3f}".format(accuracy_score(y_test, fakepred)))
