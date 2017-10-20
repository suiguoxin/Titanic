# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score

path_train = "./../data/train.csv"
path_test = "./../data/test.csv"
path_result = './../result/'

# regex_features = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*'
regex_features = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*'


# 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages_train(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_missing_ages_test(data_test, rfr):
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()

    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
    return data_test


def __set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def __set_dummies(data):
    # 特征因子化
    dummies_cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    dummies_embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    dummies_sex = pd.get_dummies(data['Sex'], prefix='Sex')
    dummies_pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')

    df = pd.concat([data, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def feature_engineering(data):
    data.loc[(data.Fare.isnull()), 'Fare'] = 0
    data = __set_cabin_type(data)
    df = __set_dummies(data)

    # scaling
    # add .values.reshape(-1, 1) to fix compilation error
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

    return df


# 用sklearn的learning_curve得到training_score和cv_score, 使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器
    title : 表格的标题
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"number of train samples")
        plt.ylabel(u"Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"score in train data set")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"score in cross-validation data set ")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def cross_validation(df):
    # 简单看看cv打分情况
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    all_data = df.filter(regex=regex_features)
    X = all_data.as_matrix()[:, 1:]
    y = all_data.as_matrix()[:, 0]
    print cross_val_score(clf, X, y, cv=5)

    # 分割数据，按照 训练数据:cv数据 = 7:3的比例
    split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
    train_df = split_train.filter(regex=regex_features)

    # 生成模型
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])

    # 对cross validation数据进行预测
    cv_df = split_cv.filter(regex=regex_features)
    predictions = clf.predict(cv_df.as_matrix()[:, 1:])

    origin_data_train = pd.read_csv("./../data/train.csv")
    bad_cases = origin_data_train.loc[
        origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]

    return bad_cases
