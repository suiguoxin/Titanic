# data analysis and wrangling
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from utile import path_train, path_test, path_result

from feature_engineering import fe, fe_title, fe_sex, fe_age, fe_embarked, fe_fare, fe_family

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

X_train, Y_train, X_test = fe(train_df, test_df)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print acc_random_forest

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})
submission.to_csv(path_result + 'kernel.csv', index=False)
