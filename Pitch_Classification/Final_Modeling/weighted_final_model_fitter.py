'''
This file contains functions on setting up/fitting a final model for my pitch prediction project, utilized balanced class weights in XGBoost modeling.
'''
#Importing packages to be used in the functions:
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.utils import class_weight
from collections import defaultdict
#Importing some individual classification and regression functions:
from location_regression_functions import *
from pitch_cat_functions import *

def final_dataframe_setup(player_name, data):
   '''
    Arguments: takes in a player name and a train/validation set dataframe of MLB league-wide pitch data. 
    Returns: a dataframe of that pitchers pitches, prepared to be fitted in modeling.
    '''
    #Filtering the full dataset for the selected player, and removing null last pitches:
    player_df = data[(data.pitcher_full_name == player_name) & (data.last_pitch_px.notnull())]

    #One-hot-encoding the necessary columns.
    ohe_cols = ['stand', 'p_throws']
    ohe_df = column_ohe_maker(player_df, ohe_cols)

    #Numerically encoding the last pitch type and the current pitch type:
    temp_df = last_pitch_type_to_num(ohe_df, 'last_pitch_type')
    output_df = pitch_type_to_num(temp_df, 'pitch_type')

    #Returning the output_df only, since splitting is not needed here:
    return output_df


def BalancedSampleWeights(y_train,class_weight_coef):
    '''
    Function for assigning sample weights to an xgboost classifier:
    '''
    classes = np.unique(y_train, axis = 0)
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key : value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]] * class_weight_coef
    sample_weights = [class_weight_dict[i] for i in y_train]
    return sample_weights

def final_xgboost_pitch_classification_fitter(dataframe):
    '''
    Arguments: takes in a training and validation dataframe of a pitcher.
    Returns: an XGBoost classification model, fitted on the data.  Utilized balanced class weights for the pitch type classes.
    '''
    #First, declaring the feature columns to use:
    xg_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'cumulative_ff_rate', 'cumulative_sl_rate', 'cumulative_ft_rate',
       'cumulative_ch_rate', 'cumulative_cu_rate', 'cumulative_si_rate',
       'cumulative_fc_rate', 'cumulative_kc_rate', 'cumulative_fs_rate',
       'cumulative_kn_rate', 'cumulative_ep_rate', 'cumulative_fo_rate',
       'cumulative_sc_rate', 'Last_Pitch_Type_Num', 'last_5_ff',
       'last_5_sl', 'last_5_ft', 'last_5_ch', 'last_5_cu', 'last_5_si',
       'last_5_fc', 'last_5_kc', 'last_5_fs', 'last_5_kn', 'last_5_ep',
       'last_5_fo', 'last_5_sc']
    
    #Setting up the X and y train/validation sets for random forest classification:
    X_xg = dataframe[xg_cols]
    y_xg = dataframe['Pitch_Type_Num']
    
    #Getting class weight coefficients:
    class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(y_xg), y = y_xg)
    weights = BalancedSampleWeights(y_xg,class_weights)
    
    #Fitting on the model, then predicting/metric scoring on validation:
    xg_model = XGBClassifier(sample_weight = weights)
    xg_model.fit(X_xg,y_xg)
 
    #returning the fit model only:
    return xg_model

def final_px_linear_regression_fitter(dataframe):
    '''
    Arguments: takes in a training and validation dataframe.
    Returns: a linear regression model of the x location of the pitch (px),fitted on the train/validation data.
    '''
    #Setting up the columns needed:
    px_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'Last_Pitch_Type_Num', 'Pitch_Type_Num']

    #Setting up the x and y inputs for linear regression:
    X_px = dataframe[px_cols]
    y_px = dataframe['px']
    
    #Fitting linear regression on the training data, then predicting and scoring on the validation set:
    lm = LinearRegression()
    lm.fit(X_px, y_px)

    return lm


def final_pz_linear_regression_fitter(dataframe):
     '''
    Arguments: takes in a training and validation dataframe.
    Returns: a linear regression model of the z location of the pitch (px),fitted on the train/validation data.
    '''
    #Setting up the columns needed:
    pz_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'Last_Pitch_Type_Num', 'Pitch_Type_Num', 'px']

    #Setting up the x and y inputs for linear regression:
    X_pz = dataframe[pz_cols]
    y_pz = dataframe['pz']


    #Fitting linear regression on the training data, then predicting and scoring on the validation set:
    lm = LinearRegression()
    lm.fit(X_pz, y_pz)

    return lm

def final_pitch_prediction_model_fitter(player_name, dataframe):
   '''
    Arguments: takes in a player name and a dataframe of MLB league-wide pitch data. 
    Returns: Runs pitch classification modeling via XGBoost for pitch types, then pitch location prediction via linear regression.  Fits models for each of the three steps for the pitcher's pitches in the dataframe, outputting a dictionary of the models.
    '''
    print('Fitting model for {}'.format(player_name)) #printing the pitcher name, so it's clear to the user

    #First, setting up the dataframe and splitting the data into train and validation sets:
    model_df = final_dataframe_setup(player_name, dataframe)

    #Classification - XGBoost:
    xg_model = final_xgboost_pitch_classification_fitter(model_df)
    
    #Linear Regression: Px then Pz:
    px_model = final_px_linear_regression_fitter(model_df)
    pz_model = final_pz_linear_regression_fitter(model_df)

    #Returning a dictionary with the player's name as key, then the 3 models in a list as the value:
    player_dict = {}
    player_dict[player_name] = [xg_model, px_model, pz_model]

    return player_dict

def final_model_multiple_pitcher_fittings(player_name_list, dataframe):
    '''
    Arguments: takes in a list of player names and a train/validation dataframe of MLB league-wide pitch data.  
    Returns: Fits an XGBoost model on the training/validation data, as well as linear regression models for pitch Px and Pz location.  
    Outputs a dictionary with the three models for each pitcher, with the pitcher name as the key.
    '''
    compiled_model_list = []
    for player in player_name_list:
        player_dict = final_pitch_prediction_model_fitter(player, dataframe)
        compiled_model_list.append(player_dict)
        
    #converting list to dict:
    model_dict = defaultdict(list)
    for model in compiled_model_list:
        for key in model:
            model_dict[key].append(model[key])
    return model_dict