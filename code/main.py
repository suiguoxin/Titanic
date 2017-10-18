# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from utile import set_missing_ages, set_cabin_type, plot_learning_curve, cross_validation

data_train = pd.read_csv("./../data/train.csv")

regex_features = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*'

# data pre-processing
# rfr to fill the missing data
data_train, rfr = set_missing_ages(data_train)
data_train = set_cabin_type(data_train)

# 特征因子化
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# scaling
# add .values.reshape(-1, 1) to fix compilation error
# delete the age and fare rows ??????????
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

# train
# 用正则取出我们要的属性值
train_df = df.filter(regex=regex_features)
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

# fit到RandomForestRegressor之中
# clf.fit(X, y)

# fit到BaggingRegressor之中
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                               bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

# learning curves
plot_learning_curve(clf, u"Learning Curve", X, y)

# cross validation
bad_cases = cross_validation(df)

# test data pre-processing, same as in train data
data_test = pd.read_csv("./../data/test.csv")
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), fare_scale_param)

# prediction
test = df_test.filter(regex=regex_features)
# predictions = clf.predict(test)
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv("./../data/logistic_regression_bagging_predictions.csv", index=False)
