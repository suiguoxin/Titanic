# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb

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


def get_oof(clf, x_train, y_train, x_test, proba=True):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)

        if proba:
            oof_train[test_index] = clf.predict_proba(x_te)[:, 1]
            oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 1]
        else:
            oof_train[test_index] = clf.predict(x_te).astype(int)
            oof_test_skf[i, :] = clf.predict(x_test).astype(int)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025,
    'probability': True,
    'random_state': SEED
}

svc = SVC(**svc_params)
knn = KNeighborsClassifier(n_neighbors=4)
# lr = LogisticRegression()
# rf = RandomForestClassifier(n_estimators=50)
# gaussian = GaussianNB()
# perceptron = Perceptron()
# test = LinearSVC()

svc_oof_train, svc_oof_test = get_oof(svc, X, y, df_test)
knn_oof_train, knn_oof_test = get_oof(knn, X, y, df_test)
# lr_oof_train, lr_oof_test = get_oof(lr, X, y, df_test)
# rf_oof_train, rf_oof_test = get_oof(rf, X, y, df_test)
# gaussian_oof_train, gaussian_oof_test = get_oof(gaussian, X, y, df_test)
# perceptron_oof_train, perceptron_oof_test = get_oof(perceptron, X, y, df_test, proba=False)
# test_oof_train, test_oof_test = get_oof(test, X, y, df_test, proba=False)

base_predictions_train = pd.DataFrame({'SVC': svc_oof_train.ravel(),
                                       'KNN': knn_oof_train.ravel(),
                                       # 'LR': lr_oof_train.ravel(),
                                       # 'RF': rf_oof_train.ravel()
                                       # 'Gaussian': gaussian_oof_train.ravel()
                                       # 'Perceptron': perceptron_oof_train.ravel()
                                       # 'Test': test_oof_train.ravel()
                                       })

base_predictions_test = pd.DataFrame({'SVC': svc_oof_test.ravel(),
                                      'KNN': knn_oof_test.ravel(),
                                      # 'LR': lr_oof_test.ravel(),
                                      # 'RF': rf_oof_test.ravel()
                                      # 'Gaussian': gaussian_oof_test.ravel()
                                      # 'Perceptron': perceptron_oof_test.ravel()
                                      # 'Test': test_oof_test.ravel()
                                      })

X = pd.concat([X, base_predictions_train], axis=1)
df_test = pd.concat([df_test, base_predictions_test], axis=1)

clf = xgb.XGBClassifier()


def model_fit(early_stopping_rounds=25):
    global clf
    cv_folds = 5
    early_stopping_rounds = early_stopping_rounds
    dtrain = xgb.DMatrix(X, label=y)
    params = clf.get_params()
    cv_result = xgb.cv(params, dtrain, num_boost_round=500, nfold=cv_folds, metrics='auc',
                       early_stopping_rounds=early_stopping_rounds)

    best_iteration = cv_result.shape[0]
    test_auc_mean = cv_result['test-auc-mean'][best_iteration - 1]
    print "Best iteration: %d" % best_iteration
    print "test auc mean: %f" % test_auc_mean
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
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=4, verbose=0)
    gsearch.fit(X, y)

    clf = gsearch.best_estimator_
    print gsearch.best_params_, gsearch.best_score_


def param_gamma():
    global clf
    param_test = {'gamma': np.arange(0, 1.1, 0.1)}
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=4, verbose=0)
    gsearch.fit(X, y)

    clf = gsearch.best_estimator_
    print gsearch.best_params_, gsearch.best_score_


def param_subsample_colsample_bytree():
    global clf
    param_test = {'subsample': np.arange(0.6, 1.05, 0.1), 'colsample_bytree': np.arange(0.6, 1.05, 0.1)}
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=4, verbose=0)
    gsearch.fit(X, y)

    clf = gsearch.best_estimator_
    print gsearch.best_params_, gsearch.best_score_


def param_reg_alpha():
    global clf
    param_test = {'reg_alpha': [1e-3, 1e-2, 0.1, 1, 100]}
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, cv=4, verbose=0)
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
result.to_csv(path_result + 'stacking_predictions.csv', index=False)
