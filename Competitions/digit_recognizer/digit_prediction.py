'''

https://www.kaggle.com/c/digit-recognizer/data

Digit Recognizer Kaggle

This approach uses SVM to classify the digits

Reference Kernel:

https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification/comments#Introduction


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import seaborn as sns
sns.set(style="darkgrid")



from sklearn.model_selection import train_test_split
from sklearn import svm



def elapsed(sec):

    '''

    This function returns the elapsed time

    '''

    if sec < 60:
        return str(round(sec)) + ' secs'

    elif sec < 3600:
        return str(round((sec) / 60)) + ' mins'

    else:
        return str(round(sec / 3600 )) + ' hrs'

    


def dataAnalysis(train):

    '''

    This function performs data anaysis by
    analysisng basic statistics and visualizing
    graphs

    '''


    # Print distribution of labels
    sns.countplot(train['label'])
    plt.title('Distribution of labels')
    plt.show()

    print('Label Count: {}'.format(train['label'].value_counts(sort = False)))



if __name__ == '__main__':
    

    start_time = time.time()

    train_df = pd.read_csv('train.csv')
    #test_img = pd.read_csv('test.csv')

    print(train_df.head())

    # Perform analysis of data
    dataAnalysis(train_df)
    

    

    
