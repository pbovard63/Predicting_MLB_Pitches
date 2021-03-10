'''
This file contains various functions to be utilized in the classification modeling process for pitch types.
'''
#Importing Packages:
import pickle
from sqlalchemy import create_engine
import pandas as pd
from importlib import reload
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def column_ohe_maker(df, column_name_list):
    '''
    Arguments: takes in a dataframe, with a list of columns in that dataframe to one-hot-encode.
    Returns: the original dataframe argument, with the original columns in column_name_list removed and the one-hot-encoded columns added on.
    '''
    for col in column_name_list:

        #Instantiating one hot encoder, and declaring column to work with:
        ohe = OneHotEncoder(drop='first', sparse=False)
        work_col = df.loc[:, [col]]

        #Fitting OHE:
        ohe_X = ohe.fit_transform(work_col)
        column_names = ohe.get_feature_names([col])
        ohe_X_df = pd.DataFrame(ohe_X, columns=column_names, index=work_col.index)

        #Adding the new one-hot encoded columns onto the dataframe:
        df = pd.concat([df, ohe_X_df], axis=1)

        #Deleting the original column that has been one-hot-encoded:
        df.drop(columns=col, inplace=True)
    
    #After looping through, returning the dataframe:
    return df

def pitch_type_to_num(df, col_name):
    '''
    Arguments: takes in a dataframe for machine learning modeling, and the column name that houses pitch types.
    Returns: a new dataframe, with the pitch types converted to a numericla coding.
    '''
    #Pulling the list of pitch types, and creating a dictionary with a number to code:
    type_list = df[col_name].value_counts().index
    type_dict = {}
    for i, pitch_type in enumerate(type_list):
        type_dict[pitch_type] = i
    print('Here is the coding for pitch type:')  #printing so user can see the codings.
    print(type_dict)
    
    #Creating a new numerical pitch type column, and assigning values:
    df['Pitch_Type_Num'] = 0
    for i, pitch in enumerate(df[col_name]):
        df.Pitch_Type_Num.iloc[i] = type_dict[pitch]
    return df
    
def random_forest_eval_kfold(Player_Name,X,y, df,k=5):
    '''
    Arguments: takes in a set of features X and a target variable y.  Y is a classification (0/1). 
    Also includes a threshold, default of 0.5, for classification purposes.  This runs K-Fold cross validation, with a default k of 5.
    Returns: Performs RandomForest classification and returns the scores.  Default classification threshold is set at 0.5.
    '''
    print('Random Forest Results for {}'.format(Player_Name))
    
    X_cv, y_cv = np.array(X), np.array(y)
    kf = KFold(n_splits=k, shuffle=True, random_state = 12)
    
    #Setting up empty lists:
    cv_rf_acc = []
    cv_rf_prec = []
    cv_rf_rec = []
    cv_rf_fbeta = []
    cv_rf_f1 = []
    
    #K-Fold Loop:
    i = 1
    for train_ind, val_ind in kf.split(X_cv,y_cv):
        X_train, y_train = X_cv[train_ind], y_cv[train_ind]
        X_val, y_val = X_cv[val_ind], y_cv[val_ind] 
    
        #Running Model and making predictions:
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_val)
        #y_pred = (rf_model.predict_proba(X_val)[:, 1] >= threshold)

        #Printing Confusion Matrix for each round:
        cm = confusion_matrix(y_val, y_pred)
        print("Confusion Matrix for Fold {}".format(i))
        print(cm)
        print('\n')
        i += 1
        
    pitch_types = df.pitch_type.value_counts().index 
    confusion_matrix_generator(cm,'{} Random Forest'.format(Player_Name), pitch_types)
    
    return rf_model

def confusion_matrix_generator(confusion_matrix, name, pitch_types):
    '''
    Arguments: takes in the basic confusion matrix, and the name of the model to title the output graph.
    Returns: a visually appealing Seaborn confusion matrix graph.
    '''
    plt.figure(dpi=150)
    sns.heatmap(confusion_matrix, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels= list(pitch_types),
           yticklabels= list(pitch_types))

    plt.xlabel('Predicted Pitch Type')
    plt.ylabel('Actual Pitch Type')
    plt.title('{} confusion matrix'.format(name));

def random_forest_pitch_pipeline(Player_Name, df, ohe_cols, feature_cols, k=5):
    '''
    Arguments: takes in a player name, a Pandas dataframe of pitch data, columns to one-hot encode, and columns to use as features. K for number of k-folds in cross validation, default=5.
    REturns: classification results of the pitchers pitch types after Random Forest classification.
    '''
    #Filtering df for the entered player:
    player_df = df[df.pitcher_full_name == Player_Name]

    #One Hot Encoding the entered columns:
    ohe_df = column_ohe_maker(player_df, ohe_cols)

    #Encoding the pitch types with numbers:
    output_df = pitch_type_to_num(ohe_df, 'pitch_type')

    #Preparing the dataframe for Random Forest Classification, using random_forest_eval_kfold function:
    model_df = output_df[output_df.last_pitch_px.notnull()]
    col = feature_cols
    X = model_df[col]
    y = model_df['Pitch_Type_Num']

    model = random_forest_eval_kfold(Player_Name,X,y, model_df,k)
    return model
