import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import re

def percent_survivor(df, column):
    """
    This function takes the dataframe and the column 
    where the EDA is to be done as input. Next it 
    calculates across unique categories of that column 
    what percentage of passengers survived.
    Finally it returns the percentage survivors.
    """
    df = df[~df.Survived.isnull()]
    df = pd.pivot_table(df, index=column, aggfunc={'Survived':'sum','PassengerId':'count'})
    df.loc[:,'Percentage_Survivors'] = df.Survived.div(df.PassengerId)
    return df.Percentage_Survivors

def one_hot_encoder(df, label):
    """
    This function converts the input label in the df into 
    one-hot-encoding format and removes the input label
    before returning the data frame.
    """
    df_ = df.copy()
    for i in df_[label].unique():
        df_.loc[:, label+'_'+i] = 0
        df_.loc[df_[label]==i, label+'_'+i]=1
    return df_.drop([label], axis=1)

class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extracts the title from the given names. Further, classifies each
    title into five unique titles: Mr, Mrs, Miss, Master, and Others."""
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        title_dict = {'Miss': ['Ms', 'Mlle'],
                      'Mrs': ['Mme'],
                      'Others':['Lady', 'the Countess', 'Jonkheer', 'Dona',
                                'Rev', 'Sir','Capt', 'Col', 'Major', 'Don', 'Dr']}
        df.loc[:,'Title'] = df.Name.apply(lambda x: re.match(pattern='.+, (.+?)\..+', string=x)[1])
        for key, values in title_dict.items():
            df.Title.replace(values, key, inplace=True)
        return df

class MissingValues(BaseEstimator, TransformerMixin):
    """Average age of the population with the same title
    is used to compute the age of the missing person."""
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        title_mean_age = {}
        for i in df.Title.unique(): title_mean_age[i] = df[df['Title']==i].Age.mean()
        df.loc[df.Age.isnull(),'Age'] = df.loc[df.Age.isnull(),'Title'].apply(lambda x: title_mean_age[x])
        df.loc[df.Fare.isnull(),'Fare']=9 #Only 1 such instance so putting a value of 9
        df.loc[df.Embarked.isnull(),'Embarked'] = 'S' # Only 2 missing values so using a random value of "S"
        return df

class FamilySize(BaseEstimator, TransformerMixin):
    """Returns SibSp + Parch"""
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.loc[:,'family_size'] = df.SibSp + df.Parch
        return df

class GenderEncoding(BaseEstimator, TransformerMixin):
    """Males represented by 0 and Females by 1"""
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.Sex.replace({'male': 0, 'female': 1}, inplace=True)
        return df

class MakeCategorical(BaseEstimator, TransformerMixin):
    """Pass the column name as an argument and convert
    the column from numerical to categorical."""
    def __init__(self, key=None):
        self.key = key

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df[self.key].replace({1:'first',2:'second',3:'third'}, inplace=True)
        return df

class FeatureCreation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        x = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','family_size']]
        for i in ['Pclass', 'Embarked','Title']:
            x = one_hot_encoder(x, i)
        return x