from joblib import load
from utilities import *
import seaborn as sns
import pandas as pd

def main():
    pipe = load('titanic.joblib')

    test = pd.read_csv('test.csv')
    answers = pd.read_csv('answers.csv')
    
    confusion_matrix = pd.crosstab(pd.Series(pipe.predict(test), name='Predicted'),
                                       pd.Series(answers.Survived.values, name='Actual'))
        
    labels = ['NotSurvived','Survived']
    sns.heatmap(confusion_matrix, annot=True, xticklabels = labels, yticklabels=labels)
    
    print('Overall accuracy: ', (confusion_matrix.iloc[0,0]+confusion_matrix.iloc[1,1])/confusion_matrix.sum().sum())

if __name__=='__main__':
    main()
