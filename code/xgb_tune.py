# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV  # Perforing grid search

from utile import path_train, path_test, path_result
from feature_engineering import fe

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

X, y, df_test = fe(train_df, test_df)

cv_folds = 5


def modelfit(alg):
    alg.fit(X, y)

    y_hat = alg.predict(X)
    y_hat_proba = alg.predict_proba(X)[:, 1]

    cv_score = cross_val_score(alg, X, y, cv=cv_folds, scoring='roc_auc')

    # Print model report:
    print "\nModel Report:"
    print "Accuracy (Train) : %.4g" % metrics.accuracy_score(y, y_hat)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_hat_proba)

    print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
        np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    return


# 0.903349
clf = xgb.XGBClassifier()
modelfit(clf)


# 0.876433 'n_estimators': 90
def param_n_estimators():
    param_test = {'n_estimators': range(20, 241, 10)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, max_depth=3, min_child_weight=1,
                                gamma=0, subsample=1,
                                colsample_bytree=1, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 max_depth=3,  min_child_weight=1
def param_min_child_max_depth():
    param_test = {'max_depth': range(1, 20, 1), 'min_child_weight': range(1, 20, 1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=90,
                                gamma=0, subsample=1,
                                colsample_bytree=1, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 gamma=0
def param_gamma():
    param_test = {'gamma': np.arange(0, 1.1, 0.1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=90,
                                max_depth=3, min_child_weight=1, subsample=1,
                                colsample_bytree=1, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 subsample=1, colsample_bytree=1
def param_subsample_colsample_bytree():
    param_test = {'subsample': np.arange(0.6, 1.05, 0.1), 'colsample_bytree': np.arange(0.6, 1.05, 0.1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=90,
                                max_depth=3, min_child_weight=1, gamma=0,
                                objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 reg_alpha = 1e-5
def param_reg_alpha():
    param_test = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=90,
                                max_depth=3, min_child_weight=1, gamma=0,
                                subsample=1, colsample_bytree=1,
                                objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# param_n_estimators()
# param_min_child_max_depth()
# param_gamma()
# param_subsample_colsample_bytree()
# param_reg_alpha()

# 0.8744795
clf_tuned = XGBClassifier(learning_rate=0.01, n_estimators=900,
                          max_depth=3, min_child_weight=1, gamma=0,
                          subsample=1, colsample_bytree=1,
                          objective='binary:logistic', nthread=-1,
                          scale_pos_weight=1, seed=0, reg_alpha=1e-5)
modelfit(clf_tuned)

# 0.79904 on submission
predictions = clf_tuned.predict(df_test)
result = pd.DataFrame({'PassengerId': test_df['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(path_result + 'xgb_predictions.csv', index=False)
