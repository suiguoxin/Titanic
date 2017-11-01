# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import GridSearchCV  # Perforing grid search

from utile import path_train, path_test, path_result
from feature_engineering import fe

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

X, y, df_test = fe(train_df, test_df)


def modelfit(alg):
    cv_folds = 5
    early_stopping_rounds = 60
    xgb_param = alg.get_params()
    xgtrain = xgb.DMatrix(X, label=y)
    cv_result = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds, metrics='auc',
                       early_stopping_rounds=early_stopping_rounds)

    alg.set_params(n_estimators=cv_result.shape[0])
    alg.fit(X, y)

    y_hat = alg.predict(X)
    y_hat_proba = alg.predict_proba(X)[:, 1]

    # Print model report:
    print "\nModel Report:"
    print "Accuracy (Train) : %.4g" % metrics.accuracy_score(y, y_hat)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_hat_proba)


# 0.9014
clf = xgb.XGBClassifier()
modelfit(clf)


# 0.8761 60
def param_n_estimators():
    param_test = {'n_estimators': range(20, 241, 10)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, max_depth=5, min_child_weight=6,
                                gamma=0, subsample=0.9,
                                colsample_bytree=0.9, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 max_depth=3,  min_child_weight=1 0.87807 7 5
def param_min_child_max_depth():
    param_test = {'max_depth': range(2, 10, 1), 'min_child_weight': range(1, 13, 1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=160,
                                gamma=0, subsample=0.9,
                                colsample_bytree=0.9, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 gamma=0.0
def param_gamma():
    param_test = {'gamma': np.arange(0, 1.1, 0.1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=160,
                                max_depth=8, min_child_weight=6, subsample=0.9,
                                colsample_bytree=0.9, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 subsample=0.9, colsample_bytree=1
def param_subsample_colsample_bytree():
    param_test = {'subsample': np.arange(0.6, 1.05, 0.1), 'colsample_bytree': np.arange(0.6, 1.05, 0.1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=160,
                                max_depth=8, min_child_weight=6, gamma=0.0,
                                objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 reg_alpha = 1e-3
def param_reg_alpha():
    param_test = {'reg_alpha': [1e-3, 1e-2, 0.1, 1, 100]}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=160,
                                max_depth=8, min_child_weight=6, gamma=0.0,
                                subsample=0.9, colsample_bytree=0.9,
                                objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# param_n_estimators()
# param_min_child_max_depth()
# param_gamma()
# param_subsample_colsample_bytree()
# param_reg_alpha()

# 0.9012
clf_tuned = XGBClassifier(learning_rate=0.01, n_estimators=1600,
                          max_depth=8, min_child_weight=6, gamma=0.0,
                          subsample=0.9, colsample_bytree=0.9,
                          objective='binary:logistic', nthread=-1,
                          scale_pos_weight=1, seed=0, reg_alpha=1e-3)
modelfit(clf_tuned)

# 0.79904 on submission
predictions = clf_tuned.predict(df_test)
result = pd.DataFrame({'PassengerId': test_df['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(path_result + 'xgb_predictions.csv', index=False)
