"""Feature engineering and model pipeline for the Titanic survival project."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLE_MAP = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Lady': 'Rare', 'the Countess': 'Rare',
             'Capt': 'Rare', 'Col': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Dr': 'Rare', 'Major': 'Rare',
             'Rev': 'Rare', 'Sir': 'Rare', 'Jonkheer': 'Rare'}

CATEGORICAL = ['Sex', 'Title', 'Embarked', 'Deck', 'FamilyType']
NUMERIC = ['Pclass', 'Age', 'FarePerPerson', 'FamilySize', 'TicketGroup', 'HasCabin', 'IsChild', 'AgeMissing']


def add_features(df, ticket_counts=None):
    """Row-level features that use no target information."""
    df = df.copy()
    title = df['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
    df['Title'] = title.replace(TITLE_MAP)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilyType'] = pd.cut(df['FamilySize'], [0, 1, 4, 20], labels=['Alone', 'Small', 'Large']).astype(str)
    df['Deck'] = df['Cabin'].fillna('U').str[0].replace({'T': 'U', 'G': 'U'})
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    counts = ticket_counts if ticket_counts is not None else df['Ticket'].value_counts()
    df['TicketGroup'] = df['Ticket'].map(counts).fillna(1)
    df['Embarked'] = df['Embarked'].fillna('S')
    df['AgeMissing'] = df['Age'].isna().astype(int)
    return df


class GroupImputer(BaseEstimator, TransformerMixin):
    """Fills Age by Title x Pclass median and Fare by Pclass median, learned on the training fold only."""

    def fit(self, X, y=None):
        self.age_ = X.groupby(['Title', 'Pclass'])['Age'].median()
        self.age_all_ = X['Age'].median()
        self.fare_ = X.groupby('Pclass')['Fare'].median()
        return self

    def transform(self, X):
        X = X.copy()
        key = pd.MultiIndex.from_frame(X[['Title', 'Pclass']])
        X['Age'] = X['Age'].fillna(pd.Series(self.age_.reindex(key).values, index=X.index)).fillna(self.age_all_)
        X['Fare'] = X['Fare'].fillna(X['Pclass'].map(self.fare_))
        X['FarePerPerson'] = np.log1p(X['Fare'] / X['TicketGroup'])
        X['IsChild'] = (X['Age'] < 14).astype(int)
        return X[CATEGORICAL + NUMERIC]
