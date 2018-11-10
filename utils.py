# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:41:19 2018

@author: abhimoha
"""

import pandas as pd
import re
import warnings
from sklearn import preprocessing

def survival_rates(all_data):
    train_data = all_data[all_data['Survived'] != 11]
    survival_rate_by_class = {}
    for i in range(1,4):
        temp = train_data[train_data['Pclass']==i]['Survived']
        survival_rate_by_class['Class '+str(i)] = temp.sum()/temp.count()
    
    survival_rate_by_sex = {}
    temp = train_data[train_data['Sex']=='male']['Survived']
    survival_rate_by_sex['male'] = temp.sum()/temp.count()
    temp = train_data[train_data['Sex']=='female']['Survived']
    survival_rate_by_sex['female'] = temp.sum()/temp.count()
    
    age_df = train_data[train_data['Age'].notnull()][['Age', 'Survived']]
    survival_rate_by_age = {}
    age_div = []
    age_diff = 10
    for i in range(0,90,age_diff):
        age_div.append((i,(i + age_diff)))
    for i in age_div:
        temp = age_df[(age_df['Age']>=i[0]) & (age_df['Age']<i[1])]['Survived']
        survival_rate_by_age[str(i[0])+'-'+str(i[1])+' years'] = temp.sum()/temp.count()
    
    survival_rate_by_sibsp = {}
    temp = train_data[train_data['SibSp']==0]['Survived']
    survival_rate_by_sibsp['zero'] = temp.sum()/temp.count()
    temp = train_data[train_data['SibSp']!=0]['Survived']
    survival_rate_by_sibsp['non_zero'] = temp.sum()/temp.count()
    
    survival_rate_by_parch = {}
    temp = train_data[train_data['Parch']==0]['Survived']
    survival_rate_by_parch['zero'] = temp.sum()/temp.count()
    temp = train_data[train_data['Parch']!=0]['Survived']
    survival_rate_by_parch['non_zero'] = temp.sum()/temp.count()
    
    survival_rate = {'Class': survival_rate_by_class, 'Sex': survival_rate_by_sex, 
                     'Age': survival_rate_by_age, 'Siblings': survival_rate_by_sibsp, 
                     'Parents/Children': survival_rate_by_parch}
    return survival_rate

def create_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    test_data.loc[:,'Survived'] = 11
    all_data = pd.concat([train_data, test_data], ignore_index = True)
    all_data.loc[:,'Title'] = all_data['Name'].str.extract('.*, (.+?)\..*', expand = False)
    mean_age_dict = {}
    for i in all_data['Title'].unique():
        mean_age_dict[i] = round(all_data[all_data['Title'] == i]['Age'].mean(),1)
    
    for i in range(len(all_data)):
        if all_data.loc[i,'Age'] != all_data.loc[i,'Age']:
            all_data.loc[i,'Age'] = mean_age_dict[all_data.loc[i,'Title']]
    return all_data

def title(name):
    pattern = '.+, (.+?)\..+'
    return re.search(pattern, name).group(1)    

def remove_null(df):
    title_mean_age = {}
    class_mean_fare = {}
    for i in df.Title.unique(): title_mean_age[i] = df[df['Title']==i].Age.mean()
    for i in df.Pclass.unique(): class_mean_fare[i] = df[df['Pclass']==i].Fare.mean()
    for i in df[df.Age.isnull()].index: df.loc[i,'Age'] = title_mean_age[df.loc[i,'Title']]
    for i in df[df.Fare.isnull()].index: df.loc[i,'Fare']=class_mean_fare[df.loc[i,'Pclass']]
    for i in df[df.Embarked.isnull()].index: df.loc[i,'Embarked'] = 'C'
    return df.drop(['Name','Cabin','Ticket'], axis = 1)

def clean():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    df = pd.concat([train_df, test_df],ignore_index=True)
    del train_df, test_df
    df['Title'] = df.Name.apply(title)
    df.Title = df.Title.replace(['Ms','Mlle'],'Miss')
    df.Title = df.Title.replace(['Mme'],'Mrs')
    df.Title = df.Title.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', \
                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    clean_df = remove_null(df)
    for i in ['Title','Sex', 'Embarked']:
        clean_df[i] = clean_df[i].replace(clean_df[i].unique(),range(len(clean_df[i].unique())))
    
    scaler = preprocessing.MinMaxScaler()
    for i in clean_df.columns:
        if i!='Survived' and i!='PassengerId':
            clean_df[i] = scaler.fit_transform(clean_df[i].values.reshape(-1,1))
    
    
    features_test = clean_df[clean_df.Survived.isnull()].drop(['Survived','PassengerId'], axis=1)
    features_train = clean_df[clean_df.Survived.notnull()].drop(['Survived','PassengerId'], axis=1)
    labels_train = clean_df[clean_df.Survived.notnull()]['Survived']
    return features_train, labels_train, features_test, pd.read_csv('answers.csv')