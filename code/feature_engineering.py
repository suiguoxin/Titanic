import pandas as pd
from collections import Counter


def fe_title_name(combine):
    title_mapping = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }

    combine['Title'] = combine.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    combine['Title'] = combine.Title.map(title_mapping)
    combine['Title'] = combine['Title'].fillna('Other')
    combine.drop('Name', axis=1, inplace=True)


def fe_age(combine):
    # iterate over Sex (0 or 1), Pclass (1, 2, 3) and titles
    # to calculate guessed values of Age for the six combinations.
    sexes = ['female', 'male']
    titles = ['Officer', 'Royalty', 'Mrs', 'Mr', 'Miss', 'Master', 'Other']
    for i, sex in enumerate(sexes):
        for j in range(0, 3):
            for k, title in enumerate(titles):
                guess_df = \
                    combine[(combine['Sex'] == sex) & (combine['Pclass'] == j + 1) & (combine['Title'] == title)][
                        'Age'].dropna()

                if guess_df.empty:
                    guess_df = \
                        combine[(combine['Sex'] == sex) & (combine['Pclass'] == j + 1)]['Age'].dropna()

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                age_guess = int(age_guess / 0.5 + 0.5) * 0.5
                combine.loc[(combine.Age.isnull()) & (combine.Sex == sex) & (combine.Pclass == j + 1) & (
                    combine['Title'] == title), 'Age'] = age_guess

    combine.loc[combine['Age'] <= 16, 'Age'] = 1
    combine.loc[(combine['Age'] > 16) & (combine['Age'] <= 32), 'Age'] = 2
    combine.loc[(combine['Age'] > 32) & (combine['Age'] <= 48), 'Age'] = 3
    combine.loc[(combine['Age'] > 48) & (combine['Age'] <= 64), 'Age'] = 4
    combine.loc[combine['Age'] > 64, 'Age'] = 5

    combine['Age'] = combine['Age'].astype(int)


def fe_embarked(combine):
    # fill the two missing values of embarked in training combine with the most common occurance
    freq_port = combine.Embarked.dropna().mode()[0]
    combine['Embarked'] = combine['Embarked'].fillna(freq_port)


def fe_fare(combine):
    combine['Fare'].fillna(combine['Fare'].dropna().median(), inplace=True)

    for idx, row in combine.iterrows():
        count = Counter(combine.Ticket)[row.Ticket]
        if count > 1:
            combine.loc[idx, 'Fare'] = row.Fare / count

    combine.loc[combine['Fare'] <= 7.775, 'Fare'] = 0
    combine.loc[(combine['Fare'] > 7.775) & (combine['Fare'] <= 13), 'Fare'] = 1
    combine.loc[combine['Fare'] > 13, 'Fare'] = 2

    combine['Fare'] = combine['Fare'].astype(int)


def fe_family(combine):
    combine['FamilySize'] = combine['SibSp'] + combine['Parch'] + 1

    combine.loc[combine['FamilySize'] == 1, 'FamilySize'] = 0
    combine.loc[(combine['FamilySize'] >= 2) & (combine['FamilySize'] <= 4), 'FamilySize'] = 1
    combine.loc[combine['FamilySize'] >= 5, 'FamilySize'] = 2

    combine.drop(['Parch', 'SibSp'], axis=1, inplace=True)


def fe_dummy(combine):
    return pd.get_dummies(combine, columns=[
        'Age',
        'Cabin',
        'Embarked',
        'Ticket',
        'Fare',
        'Sex',
        'Pclass',
        'Title',
        'FamilySize'
    ])


def fe(train_df, test_df):
    y = train_df["Survived"]
    train_df.drop("Survived", 1, inplace=True)
    train_df['TrainTest'] = 1
    test_df['TrainTest'] = 0

    combine = train_df.append(test_df)
    combine.reset_index(drop=True, inplace=True)

    # combine.drop('Cabin', axis=1, inplace=True)
    combine['Cabin'] = combine['Cabin'].str[0]
    combine['Cabin'] = combine['Cabin'].fillna('U')

    fe_fare(combine)
    combine['Ticket'] = combine['Ticket'].str[0]
    # combine.drop('Ticket', axis=1, inplace=True)

    fe_title_name(combine)
    fe_age(combine)
    fe_family(combine)
    fe_embarked(combine)
    combine = fe_dummy(combine)

    train_df = combine[combine['TrainTest'] == 1].drop(['TrainTest'], axis=1)
    test_df = combine[combine['TrainTest'] == 0].drop(['TrainTest'], axis=1)
    test_df.reset_index(drop=True, inplace=True)

    # train_df.info()
    # test_df.info()
    X = train_df.drop('PassengerId', axis=1)
    X_test = test_df.drop('PassengerId', axis=1)

    return X, y, X_test
