'''

Titanic: Predict the survivors

Optimization and EDA

https://www.kaggle.com/c/titanic/rules


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import seaborn as sns
sns.set(style="darkgrid")


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics




def elapsed(sec):

      '''

      This function returns the elapsed time

      '''

      if sec < 60:
            return str(round(sec)) +  ' secs'

      elif sec < 3600:
            return str(round((sec) / 60)) + ' mins'

      else:
            return str(round(sec / 3600)) + ' hrs'



def dispMissingVals(df):

    
    '''

      This function displays the missing values (NaN or Nulls)
      in the data sets

    '''

    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col , df[col].isnull().sum()))


    print('=============================')
    print()
    

def dataAnalysis(train , test):

    '''

    This function does data analysis and plotting
    of the dataset to understand the features better

    '''

    target = train['Survived']
    #sns.countplot(x = 'Sex' , hue = 'Survived' , data = train)

    fig , ax = plt.subplots(1 , 3 , figsize = (10 , 6))
    a = sns.countplot(x = 'Sex' , data = train , ax = ax[0] , order = ['male' , 'female'])
    b = sns.countplot(x = 'Sex' , data = train[target == 1] , ax = ax[1] , order = ['male' , 'female'])
    c = sns.countplot(x = 'Sex' , data= train[ ((train['Age'] < 21) & (train['Survived'] == 1)) ] , order = [1 , 2 , 3])

    ax[0].set_title('All passengers')
    ax[1].set_title('Survived Passenger')
    ax[2].set_title('Survived under age 21')
    
    #plt.show()



def featureEngineering(train , test):


    '''

    This function perfroms selection of relevant features
    handles missing or null values


    '''
    print('Missing values in training set')
    dispMissingVals(train)

    print('Missing values in test set')
    dispMissingVals(test)

    # Create new column deck from Cabin
    train['Deck'] = train['Cabin'].str.get(0)
    test['Deck'] = test['Cabin'].str.get(0)

    # Fill the missing values as NA
    train['Deck'] = train['Deck'].fillna('NA')
    test['Deck'] = test['Deck'].fillna('NA')

    '''

    fig , ax = plt.subplots(1 , 2 , figsize = (8 , 5))

    ta = sns.countplot(x = 'Deck' , data = train , ax = ax[0])
    tb = sns.countplot(x = 'Deck' , data = test , ax = ax[1])

    ax[0].set_title('Train Deck')
    ax[1].set_title('test Deck')

    plt.show()

    '''

    # Replace T with G
    train['Deck'].replace('T' , 'G' , inplace = True)

    # Drop the cabin column
    train.drop('Cabin' , axis = 1 , inplace = True)
    test.drop('Cabin' , axis = 1 , inplace = True)

    # Fill the embarked missing values with S - the most common occuring value
    train.loc[train['Embarked'].isna() , 'Embarked'] = 'S'

    # Fill the age missing values
    age_fill = train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median()

    print(age_fill)
    print()

    for cl in range(1 , 4):
        for sex in ['male' , 'female']:
            for em in ['C' , 'Q' , 'S']:

                val = pd.to_numeric(age_fill.xs(cl).xs(sex).xs(em).Age)

                train.loc[(train['Age'].isna() & (train['Pclass'] == cl)
                          & (train['Sex'] == sex) & (train['Embarked'] == em)), 'Age'] = val



                test.loc[(train['Age'].isna() & (train['Pclass'] == cl)
                          & (train['Sex'] == sex) & (train['Embarked'] == em)), 'Age'] = val                


    print()
    print(train.groupby(['Pclass' , 'Sex' , 'Embarked'])[['Age']].median())

    #dispMissingVals(train)
    #dispMissingVals(test)
    
    

if __name__ == '__main__':

    start_time = time.time()


    # Load the dataset
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')    
    
    # Analyse data
    dataAnalysis(train_df , test_df)

    # Handle Missing vals and drop cols
    featureEngineering(train_df , test_df)




    
    elapsed_time = elapsed(time.time() - start_time)
    print('Elapsed time: ' , elapsed_time)  
