# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from matplotlib.pylab import rcParams

from utile import path_train, path_test, path_result, regex_features
from utile import set_missing_ages_train, set_missing_ages_test, feature_engineering

from feature_engineering import fe

rcParams['figure.figsize'] = 12, 4

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

# feature engineering
# data_train, rfr = set_missing_ages_train(train_df)
# df_train = feature_engineering(data_train)
#
# # train
# train_np = df_train.filter(regex=regex_features).as_matrix()
# y = train_np[:, 0]
# X = train_np[:, 1:]

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


# 0.907698
gbm = GradientBoostingClassifier(random_state=10)
modelfit(gbm)


# 0.87167 'n_estimators': 40
def param_n_estimators():
    param_test = {'n_estimators': range(20, 241, 10)}
    gsearch = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=3, min_samples_leaf=1,
                                             min_samples_split=2, subsample=1.0,
                                             random_state=10),
        param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87243 max_depth=3,  min_samples_split=8
def param_min_samples_max_depth():
    param_test = {'max_depth': range(1, 20, 1), 'min_samples_split': range(2, 20, 1)}
    gsearch = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=40, subsample=1.0,
                                             random_state=10),
        param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87243 max_features = 8
def param_max_features():
    param_test = {'max_features': range(1, 9, 1)}
    gsearch = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=40, max_depth=3, min_samples_split=8,
                                             subsample=1.0, random_state=10),
        param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# 0.87319 subsample = 0.9
def param_subsample():
    param_test = {'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]}
    gsearch = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=40, max_depth=3, min_samples_split=8,
                                             random_state=10, max_features=8),
        param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(X, y)
    print gsearch.best_params_, gsearch.best_score_


# param_n_estimators()
# param_min_samples_max_depth()
# param_max_features()
# param_subsample()

# 0.897171
gbm_tuned = GradientBoostingClassifier(learning_rate=0.01, n_estimators=400, max_depth=3, min_samples_split=8,
                                       subsample=0.9, random_state=10, max_features=8)
modelfit(gbm_tuned)

# feat_imp = pd.Series(gbm0.feature_importances_).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')
# plt.show()

# # test data pre-processing, same as in train data
# data_test = set_missing_ages_test(test_df, rfr)
#
# # feature engineering
# df_test = feature_engineering(data_test)
# # 用正则取出我们要的属性值
# df_test = df_test.filter(regex=regex_features)

# prediction
# predictions = clf.predict(test)
# 0.79904 on submission
predictions = gbm_tuned.predict(df_test)
result = pd.DataFrame({'PassengerId': test_df['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(path_result + 'gbm_predictions.csv', index=False)
