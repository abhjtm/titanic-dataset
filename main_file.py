import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from utilities import *
from joblib import dump

def main():
    train_df = pd.read_csv('train.csv')
    pipe = make_pipeline(TitleExtractor(), 
                         MissingValues(), 
                         FamilySize(), 
                         GenderEncoding(),
                         MakeCategorical('Pclass'), 
                         FeatureCreation(), 
                         MinMaxScaler(), 
                         PCA(),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=40))
    pipe.fit(train_df.copy(), train_df.Survived.values)
    dump(pipe, 'titanic.joblib')

if __name__=='__main__':
    main()
