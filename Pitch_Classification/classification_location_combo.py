'''
This file contains functions that utilize classification and regression functions in conjunction to predict pitch type and location of an MLB pitch.
'''
#Importing packages to be used in the functions:
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import scipy.stats as st
import matplotlib.pyplot as plt
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

#Importing some individual classification and regression functions:
from location_regression_functions import *
from pitch_cat_functions import *

def pitch_type_regression_setup(player_name, full_dataframe, feature_cols):
    '''
    Function to create pitch type regression for a given player, from a dataset.
    Returns: a pandas dataframes with a predicted pitch type as a new column utilizing RandomForestRegression.
    '''
    #Filtering the given dataframe for the pitcher's pitches only:
    df = full_dataframe[full_dataframe.pitcher_full_name == player_name]
    
    #One-Hot-Encoding the OHE columns in ohe_cols:
    ohe_cols = ['stand', 'p_throws']
    ohe_df = column_ohe_maker(df, ohe_cols)

    #Encoding the pitch types with numbers, first with the previous pitch type, then the current pitch type:
    #temp_df = last_pitch_type_to_num(ohe_df, 'last_pitch_type')
    output_df = pitch_type_to_num(ohe_df, 'pitch_type')

    #Preparing the dataframe for Random Forest Classification, using random_forest_eval_kfold function:
    model_df = output_df[output_df.last_pitch_px.notnull()]
    return model_df


def pitch_type_regression_rf(player_name, full_dataframe, feature_cols):
    '''
    Takes in a player name, Pandas dataframe for modeling, and a list of feature columns.
    REturns model.
    '''
    #Running the setup function to get the dataframe for modeling:
    model_df = pitch_type_regression_setup(player_name, full_dataframe, feature_cols)
    X = model_df[feature_cols]
    y = model_df['Pitch_Type_Num']

    print('Random Forest Results for {}'.format(player_name))
    
    #Setting up arrays for CV:
    X_cv, y_cv = np.array(X), np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state = 12)
    
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

        #Printing Confusion Matrix for each round:
        cm = confusion_matrix(y_val, y_pred)
        print("Confusion Matrix for Fold {}".format(i))
        print(cm)
        print('\n')
        i += 1

        #Adding the predictions to the dataframe, to use for regression:
        reg_df = pd.DataFrame(X_val, columns=X.columns)
        print(reg_df.columns)
        reg_df['pred_pitch_type'] = y_pred
        px_X = reg_df.drop(columns=['px'])
        px_Y = reg_df.px

        split_and_train_val_simple_lr_w_cv(px_X, px_y)

    return rf_model








