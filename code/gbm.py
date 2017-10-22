# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV  # Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

from utile import path_train, path_test, path_result, regex_features
from utile import set_missing_ages_train, set_missing_ages_test, plot_learning_curve, cross_validation, \
    feature_engineering

rcParams['figure.figsize'] = 12, 4

data_train = pd.read_csv(path_train)
data_test = pd.read_csv(path_test)

# feature engineering
data_train, rfr = set_missing_ages_train(data_train)
df_train = feature_engineering(data_train)

# train
train_np = df_train.filter(regex=regex_features).as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]

cv_folds = 5

gbm0 = GradientBoostingClassifier(random_state=10)

cv_score = cross_val_score(gbm0, X, y, cv=cv_folds, scoring='roc_auc')

print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

gbm0.fit(X, y)

# test data pre-processing, same as in train data
data_test = set_missing_ages_test(data_test, rfr)

# feature engineering
df_test = feature_engineering(data_test)
# 用正则取出我们要的属性值
df_test = df_test.filter(regex=regex_features)

# prediction
# predictions = clf.predict(test)
predictions = gbm0.predict(df_test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(path_result + 'gbm_predictions.csv', index=False)
