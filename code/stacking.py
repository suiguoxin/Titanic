# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from utile import path_train, path_test, path_result
from feature_engineering import fe

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

X, y, df_test = fe(train_df, test_df)

# Some useful parameters which will come in handy later on
ntrain = X.shape[0]
ntest = df_test.shape[0]
SEED = 0  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)[:, 1]
        oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 1]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025,
    'probability': True
}

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
knn = KNeighborsClassifier(n_neighbors=3)
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100)
gaussian = GaussianNB()

svc_oof_train, svc_oof_test = get_oof(svc, X, y, df_test)
knn_oof_train, knn_oof_test = get_oof(knn, X, y, df_test)
lr_oof_train, lr_oof_test = get_oof(lr, X, y, df_test)
rf_oof_train, rf_oof_test = get_oof(rf, X, y, df_test)
gaussian_oof_train, gaussian_oof_test = get_oof(gaussian, X, y, df_test)

base_predictions_train = pd.DataFrame({'SVC': svc_oof_train.ravel(),
                                       'KNN': knn_oof_train.ravel(),
                                       'LR': lr_oof_train.ravel(),
                                       'RF': rf_oof_train.ravel(),
                                       'Gaussian': gaussian_oof_train.ravel()
                                       })

base_predictions_test = pd.DataFrame({'SVC': svc_oof_test.ravel(),
                                      'KNN': knn_oof_test.ravel(),
                                      'LR': lr_oof_test.ravel(),
                                      'RF': rf_oof_test.ravel(),
                                      'Gaussian': gaussian_oof_test.ravel()
                                      })

X = pd.concat([X, base_predictions_train], axis=1)
df_test = pd.concat([df_test, base_predictions_test], axis=1)


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
    param_test = {'n_estimators': range(10, 40, 5)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=5,
                                gamma=0, subsample=0.9,
                                colsample_bytree=0.9, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 max_depth=7,  min_child_weight=2
def param_min_child_max_depth():
    param_test = {'max_depth': range(1, 10, 1), 'min_child_weight': range(1, 13, 1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=20,
                                gamma=0, subsample=0.9,
                                colsample_bytree=0.9, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 gamma=0.2
def param_gamma():
    param_test = {'gamma': np.arange(0, 1.1, 0.1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=25,
                                max_depth=7, min_child_weight=2, subsample=0.9,
                                colsample_bytree=0.9, objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 subsample=0.6, colsample_bytree=0.9
def param_subsample_colsample_bytree():
    param_test = {'subsample': np.arange(0.6, 1.05, 0.1), 'colsample_bytree': np.arange(0.6, 1.05, 0.1)}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=25,
                                max_depth=7, min_child_weight=2, gamma=0.2,
                                objective='binary:logistic', nthread=-1,
                                scale_pos_weight=1, seed=0),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87643 reg_alpha = 1e-2
def param_reg_alpha():
    param_test = {'reg_alpha': [1e-3, 1e-2, 0.1, 1, 100]}
    gsearch = GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=25,
                                max_depth=7, min_child_weight=2, gamma=0.2,
                                subsample=0.7, colsample_bytree=0.8,
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
clf_tuned = XGBClassifier(learning_rate=0.01, n_estimators=250,
                          max_depth=7, min_child_weight=2, gamma=0.2,
                          subsample=0.7, colsample_bytree=0.8,
                          objective='binary:logistic', nthread=-1,
                          scale_pos_weight=1, seed=0, reg_alpha=1e-2)
modelfit(clf_tuned)

# 0.79904 on submission
predictions = clf.predict(df_test)
result = pd.DataFrame({'PassengerId': test_df['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(path_result + 'ensemble_predictions.csv', index=False)
