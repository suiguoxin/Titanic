# -*- coding: utf-8 -*-
import csv
import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from utile import path_train, path_test, path_result

with open(path_train, 'rt') as f:
    data_train = list(csv.DictReader(f))

data_test = pd.read_csv("./../data/test.csv")  # data frame
with open(path_test, 'rt') as f:
    dtest = list(csv.DictReader(f))  # a list of dict

_all_xs = [{k: v for k, v in row.items() if k != 'Survived'} for row in data_train]
_all_ys = np.array([int(row['Survived']) for row in data_train])

all_xs, all_ys = shuffle(_all_xs, _all_ys, random_state=0)


def feature_engineering(data):
    for x in data:
        if x['Age']:
            x['Age'] = float(x['Age'])
        else:
            x.pop('Age')

        if not x['Fare']:
            x['Fare'] = 0
        else:
            x['Fare'] = float(x['Fare'])

        x['SibSp'] = int(x['SibSp'])
        x['Parch'] = int(x['Parch'])

        return

feature_engineering(all_xs)
feature_engineering(dtest)

train_xs, valid_xs, train_ys, valid_ys = train_test_split(
    all_xs, all_ys, test_size=0.25, random_state=0)
print('{} items total, {:.1%} true'.format(len(all_xs), np.mean(all_ys)))


class CSCTransformer:
    def transform(self, xs):
        # work around https://github.com/dmlc/xgboost/issues/1238#issuecomment-243872543
        return xs.tocsc()

    def fit(self, *args):
        return self

clf = xgb.XGBClassifier()
vec = DictVectorizer()
pipeline = make_pipeline(vec, CSCTransformer(), clf)


def evaluate(_clf):
    scores = cross_val_score(_clf, all_xs, all_ys, scoring='accuracy', cv=10)
    print('Accuracy: {:.3f} ± {:.3f}'.format(np.mean(scores), 2 * np.std(scores)))
    _clf.fit(train_xs, train_ys)  # so that parts of the original pipeline are fitted


def train(_clf):
    _clf.fit(all_xs, all_ys)


def predict(_clf):
    predictions = _clf.predict(dtest)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv(path_result + 'xgboost_predictions.csv', index=False)


# evaluate(pipeline)
train(pipeline)
predict(pipeline)
