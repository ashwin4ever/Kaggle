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
import os
import warnings
warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , cross_val_score


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

    
def drawImage(df , num):


    '''

    This function plots the MNIST digits 

    '''
    img_arr = df[df['label'] == num].values
    print(img_arr.shape)
    #img_arr = df.iloc[num , 1 : ]
    img_arr = img_arr[0 , 1 :].reshape(28 , 28)
    plt.imshow(img_arr)
    plt.show()

    
    

def dataAnalysis(train):

    '''

    This function performs data anaysis by
    analysisng basic statistics and visualizing
    graphs

    '''

    '''

    # Print distribution of labels
    sns.countplot(train['label'])
    plt.title('Distribution of labels')
    plt.show()

    '''

    print('Label Count: {}'.format(train['label'].value_counts(sort = False)))



    # Plot histogram to see the range of pixel values
    # Viewing any random number
    plt.hist(train.iloc[1 , 1 : ])
    plt.show()


def featureEngineering(train):

    '''

    This function performs selection of features
    and handles missing,null values if applicable

    Return the target label and train data

    '''

    # Get the target label
    target = train['label'].values

    # Drop the target column from the train data
    train.drop('label' , axis = 1 , inplace = True)


    # Split into train and test
    train_img , test_img , train_lbl , test_lbl = train_test_split(train ,
                                                                   target ,
                                                                   train_size = 0.8 ,
                                                                   random_state = 0)
    


    return train_img , test_img , train_lbl , test_lbl
    

def applyML(train_img , test_img , train_lbl , test_lbl , test_df):


    '''

    This function applies ML algorithms
    and predicts the result classes

    '''

    train_img[train_img > 0] = 1
    test_img[test_img > 0] = 1

    test_df[test_df > 0] = 1

    
    # Create a SVM classifier
    svm_clf = SVC(C = 4 , gamma = 'scale')

    svm_cv_score = cross_val_score(svm_clf , train_img , train_lbl.ravel() , cv = 6)
    print('SVM: {}'.format(svm_cv_score))
    
    

    # Predict using SVM
    svm_clf.fit(train_img , train_lbl.ravel())
    svm_pred = svm_clf.predict(test_img)


    # Predict for test data
    results = svm_clf.predict(test_df)

    df_res = pd.DataFrame(results)
    df_res.index.name = 'ImageId'
    df_res.columns = ['Label']
    df_res.index += 1
    df_res.to_csv('results.csv' , header = True)
    
    


if __name__ == '__main__':
    

    start_time = time.time()

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(train_df.head())

    # View Plots
    num = 4
    #drawImage(train_df , num)

    # Perform analysis of data
    dataAnalysis(train_df)

    train_img , test_img , train_lbl , test_lbl = featureEngineering(train_df)

    

    applyML(train_img , test_img , train_lbl , test_lbl , test_df)

    elapsed_time = time.time() - start_time
    print('Elapsed Time: ' , elapsed(elapsed_time))
    

    

    
