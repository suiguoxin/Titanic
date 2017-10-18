# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from utile import set_missing_ages_train, set_missing_ages_test, plot_learning_curve, cross_validation, \
    feature_engineering

path_result = './../result/'

data_train = pd.read_csv("./../data/train.csv")
data_test = pd.read_csv("./../data/test.csv")

regex_features = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*'

# data pre-processing
# feature engineering
# TODO: delete port, discrete age
data_train, rfr = set_missing_ages_train(data_train)
df_train = feature_engineering(data_train)

# train
train_np = df_train.filter(regex=regex_features).as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

# fit到RandomForestRegressor之中
# clf.fit(X, y)
# fit到BaggingRegressor之中
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                               bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

# learning curves
plot_learning_curve(bagging_clf, u"Learning Curve", X, y)

# cross validation
bad_cases = cross_validation(df_train)

# test data pre-processing, same as in train data
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
data_test = set_missing_ages_test(data_test, rfr)

# feature engineering
df_test = feature_engineering(data_test)
# 用正则取出我们要的属性值
df_test= df_test.filter(regex=regex_features)

# prediction
# predictions = clf.predict(test)
predictions = bagging_clf.predict(df_test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(path_result + 'logistic_regression_bagging_predictions.csv', index=False)
