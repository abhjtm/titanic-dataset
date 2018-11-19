import pandas as pd
def percent_survivor(df, index):
    """
    This function takes the dataframe and the column 
    where the EDA is to be done as input. Next it 
    calculates across unique categories of that column 
    what percentage of passengers survived.
    Finally it returns the percentage survivors.
    """
    df = pd.pivot_table(df, index=index, aggfunc={'Survived':'sum','PassengerId':'count'})
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