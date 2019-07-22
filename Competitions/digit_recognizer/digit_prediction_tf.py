'''

https://www.kaggle.com/c/digit-recognizer/data

Digit Recognizer Kaggle

This approach uses SVM to classify the digits

Reference:
https://www.kaggle.com/ndalziel/beginner-s-guide-to-classification-tensorflow/notebook


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import os
import warnings
warnings.filterwarnings('ignore')


import seaborn as sns
sns.set(style="darkgrid")

import tensorflow as tf
from tensorflow.python.framework import ops


def elapsed(sec):

    '''

    This function returns the elapsed time

    '''


    if sec < 60:
        return str(round(sec)) + ' secs'


    elif sec < 3600:
        return str(round(sec / 60)) + ' mins'

    else:
        return str(round(sec / 3600)) + ' hrs'




def plotImage(df , n):

    '''

    This funcion plots the MNIST images given by n

    '''

    labels = df['label']

    print(labels)

    for i in range(1 , n):

        plt.subplot(1 , 10 , i)
        img_arr = df.iloc[i , 1 : ].values.reshape(28 , 28)
        plt.imshow(img_arr)
        plt.title(labels[i])


    plt.show()


def featureEngineering(train , test):

    '''

    This function performs selection of features
    and handles missing,null values if applicable

    Return the target label and train data

    '''

    x_train = train.iloc[ : , 1 : ]
    y_train = train['label'].values

    # rows = 42000 , cols = 784
    r , c = x_train.shape


    print(x_train.shape)

    # Split into trainin and Cross Val set
    # Use 40000 records for training
    # Use 2000 records for CV

    # Train data
    x_tr = x_train[ : 40000].T.values
    y_tr = y_train[ : 40000]
    y_tr = pd.get_dummies(y_tr).T.values

    # print(y_tr , y_tr.shape)

    # CV data
    x_tr_cv = x_train[40000 : 4200000].T.values
    y_tr_cv = y_train[40000 : 420000]
    y_tr_cv = pd.get_dummies(y_tr_cv).T.values



    



if __name__ == '__main__':

    start_time = time.time()

    # Read train data
    train = pd.read_csv('train.csv')

    # read test data
    test = pd.read_csv('test.csv')

    #plotImage(train , 10)

    featureEngineering(train , test)


    elapsed_time = time.time() - start_time
    print('Elapsed Time: ' , elapsed(elapsed_time))
    

    
