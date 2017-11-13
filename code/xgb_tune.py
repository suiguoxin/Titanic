# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import metrics  # Additional sklearn functions
from sklearn.model_selection import GridSearchCV  # Performing grid search

from utile import path_train, path_test, path_result
from feature_engineering import fe

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

X, y, df_test = fe(train_df, test_df)
clf = xgb.XGBClassifier()


def model_fit(early_stopping_rounds=25):
    global clf
    cv_folds = 5
    early_stopping_rounds = early_stopping_rounds
    dtrain = xgb.DMatrix(X, label=y)
    params = clf.get_params()
    cv_result = xgb.cv(params, dtrain, num_boost_round=600, nfold=cv_folds, metrics='auc',
                       early_stopping_rounds=early_stopping_rounds)

    best_iteration = cv_result.shape[0]
    test_auc_mean = cv_result['test-auc-mean'][best_iteration - 1]
    # test_error_mean = cv_result['test-error-mean'][best_iteration - 1]
    print "Best iteration: %d" % best_iteration
    print "test auc mean: %f" % test_auc_mean
    # print "test error mean: %f" % test_error_mean
    clf.set_params(n_estimators=best_iteration)
    clf.fit(X, y)

    y_hat = clf.predict(X)
    y_hat_proba = clf.predict_proba(X)[:, 1]
    print "\nModel Report:"
    print "Accuracy (Train) : %.4g" % metrics.accuracy_score(y, y_hat)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_hat_proba)


def param_min_child_max_depth():
    global clf
    param_test = {'max_depth': range(1, 10, 1), 'min_child_weight': range(1, 13, 1)}
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=4, verbose=1)
    gsearch.fit(X, y)

    clf = gsearch.best_estimator_
    print gsearch.best_params_, gsearch.best_score_


def param_gamma():
    global clf
    param_test = {'gamma': np.arange(0, 1.1, 0.1)}
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=4, verbose=1)
    gsearch.fit(X, y)

    clf = gsearch.best_estimator_
    print gsearch.best_params_, gsearch.best_score_


def param_subsample_colsample_bytree():
    global clf
    param_test = {'subsample': np.arange(0.6, 1.05, 0.1), 'colsample_bytree': np.arange(0.6, 1.05, 0.1)}
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=4, verbose=1)
    gsearch.fit(X, y)

    clf = gsearch.best_estimator_
    print gsearch.best_params_, gsearch.best_score_


def param_reg_alpha():
    global clf
    param_test = {'reg_alpha': [1e-3, 1e-2, 0.1, 1, 100]}
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=4, verbose=1)
    gsearch.fit(X, y)

    clf = gsearch.best_estimator_
    print gsearch.best_params_, gsearch.best_score_


model_fit()
param_min_child_max_depth()
param_gamma()
param_subsample_colsample_bytree()
param_reg_alpha()

clf.set_params(learning_rate=0.01)
model_fit(early_stopping_rounds=35)

predictions = clf.predict(df_test)
result = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].as_matrix(),
    'Survived': predictions.astype(np.int32)
})
result.to_csv(path_result + 'xgb_predictions.csv', index=False)
