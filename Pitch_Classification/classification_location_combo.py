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
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor 


#Importing some individual classification and regression functions:
from location_regression_functions import *
from pitch_cat_functions import *

def split_pitch_data(df, test_size=.2, seed=32):

    rs = np.random.RandomState(seed)
    
    # Here, we select a sample (`choice`) from all possible unique users
    at_bats = df['ab_id'].unique()
    val_abs = rs.choice(at_bats, 
                           size=int(at_bats.shape[0] * test_size), 
                           replace=False)

    df_tr = df[~df['ab_id'].isin(val_abs)]
    df_val = df[df['ab_id'].isin(val_abs)] 
    
    print(f"Actual Test Size: {df_val.shape[0] / df_tr.shape[0]}")
    
    return df_tr, df_val

def dataframe_setup(player_name, data, split_size = 0.2):
    '''
    Arguments: takes in a player name and a dataframe of MLB league-wide pitch data.  The default split size for train/validation by at bat is 0.20.
    Returns: a dataframe of that pitchers pitches, split into training and validation sets for modeling.
    '''
    #Filtering the full dataset for the selected player, and removing null last pitches:
    player_df = data[(data.pitcher_full_name == player_name) & (data.last_pitch_px.notnull())]

    #One-hot-encoding the necessary columns.
    ohe_cols = ['stand', 'p_throws']
    ohe_df = column_ohe_maker(player_df, ohe_cols)

    #Numerically encoding the last pitch type and the current pitch type:
    temp_df = last_pitch_type_to_num(ohe_df, 'last_pitch_type')
    output_df = pitch_type_to_num(temp_df, 'pitch_type')

    #Splitting data into the training and validation dataframe:
    training_pitches, val_pitches = split_pitch_data(output_df, test_size=split_size)
    return training_pitches, val_pitches

def random_forest_pitch_classification(train_df, val_df):
    '''
    Arguments: takes in a training and validation dataframe of a pitcher.
    Returns: a random forest classification model, with metrics on the validation set.
    '''
    #First, declaring the feature columns to use:
    rf_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'cumulative_ff_rate', 'cumulative_sl_rate', 'cumulative_ft_rate',
       'cumulative_ch_rate', 'cumulative_cu_rate', 'cumulative_si_rate',
       'cumulative_fc_rate', 'cumulative_kc_rate', 'cumulative_fs_rate',
       'cumulative_kn_rate', 'cumulative_ep_rate', 'cumulative_fo_rate',
       'cumulative_sc_rate', 'Last_Pitch_Type_Num']
    
    #Setting up the X and y train/validation sets for random forest classification:
    X_rf_train = train_df[rf_cols]
    y_rf_train = train_df['Pitch_Type_Num']
    X_rf_val = val_df[rf_cols]
    y_rf_val = val_df['Pitch_Type_Num']

    #Fitting on the model, then predicting/metric scoring on validation:
    rf_model = RandomForestClassifier()
    rf_model.fit(X_rf_train,y_rf_train)
    y_rf_pred = rf_model.predict(X_rf_val)
    #Metrics:
    acc = accuracy_score(y_rf_val, y_rf_pred)
    prec = precision_score(y_rf_val, y_rf_pred, average='macro'), 
    recall = recall_score(y_rf_val, y_rf_pred, average='macro')
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(prec))
    print('Recall: {}'.format(recall))

    #Confusion Matrix
    cm = confusion_matrix(y_rf_val, y_rf_pred)
    print('Random Forest Pitch Classification confusion matrix results:')
    print(cm)

    #Mapping the predictions onto the validation dataframe and outputting:
    val_df['pitch_pred'] = y_rf_pred

    #Mapping out prediction probabilities:
    y_probs = rf_model.predict_proba(X_rf_val)
    pitch_type_list = val_df.pitch_type.value_counts().index
    for i, pitch in enumerate(pitch_type_list):
        col_name = pitch + '_prob'
        val_df[col_name] = y_probs[:,i]

    return rf_model, val_df

def xgboost_pitch_classification(train_df, val_df):
    '''
    Arguments: takes in a training and validation dataframe of a pitcher.
    Returns: an XGBoost classification model, with metrics on the validation set.
    '''
    #First, declaring the feature columns to use:
    xg_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'cumulative_ff_rate', 'cumulative_sl_rate', 'cumulative_ft_rate',
       'cumulative_ch_rate', 'cumulative_cu_rate', 'cumulative_si_rate',
       'cumulative_fc_rate', 'cumulative_kc_rate', 'cumulative_fs_rate',
       'cumulative_kn_rate', 'cumulative_ep_rate', 'cumulative_fo_rate',
       'cumulative_sc_rate', 'Last_Pitch_Type_Num']
    
    #Setting up the X and y train/validation sets for random forest classification:
    X_xg_train = train_df[xg_cols]
    y_xg_train = train_df['Pitch_Type_Num']
    X_xg_val = val_df[xg_cols]
    y_xg_val = val_df['Pitch_Type_Num']

    #Fitting on the model, then predicting/metric scoring on validation:
    xg_model = XGBClassifier()
    xg_model.fit(X_xg_train,y_xg_train)
    y_xg_pred = xg_model.predict(X_xg_val)
    #Metrics:
    acc = accuracy_score(y_xg_val, y_xg_pred)
    prec = precision_score(y_xg_val, y_xg_pred, average='macro'), 
    recall = recall_score(y_xg_val, y_xg_pred, average='macro')
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(prec))
    print('Recall: {}'.format(recall))

    #Confusion Matrix
    cm = confusion_matrix(y_xg_val, y_xg_pred)
    print('XGBoost Pitch Classification confusion matrix results:')
    print(cm)

    #Mapping the predictions onto the validation dataframe and outputting:
    val_df['pitch_pred'] = y_xg_pred

    #Mapping out prediction probabilities:
    y_probs = xg_model.predict_proba(X_xg_val)
    pitch_type_list = val_df.pitch_type.value_counts().index
    for i, pitch in enumerate(pitch_type_list):
        col_name = pitch + '_prob'
        val_df[col_name] = y_probs[:,i]

    return xg_model, val_df


def px_linear_regression(train_df, val_df):
    '''
    Arguments: takes in a training and validation dataframe.
    Returns: a linear regression model of the x location of the pitch (px), with the predicted values mapped onto the validation dataframe input.
    '''
    #Setting up the columns needed:
    px_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'cumulative_ff_rate', 'cumulative_sl_rate', 'cumulative_ft_rate',
       'cumulative_ch_rate', 'cumulative_cu_rate', 'cumulative_si_rate',
       'cumulative_fc_rate', 'cumulative_kc_rate', 'cumulative_fs_rate',
       'cumulative_kn_rate', 'cumulative_ep_rate', 'cumulative_fo_rate',
       'cumulative_sc_rate', 'Last_Pitch_Type_Num', 'Pitch_Type_Num']
    px_cols_val = px_cols.copy()
    px_cols_val.append('pitch_pred')
    px_cols_val.remove('Pitch_Type_Num')

    #Setting up the x and y inputs for linear regression:
    X_px_tr = train_df[px_cols]
    y_px_tr = train_df['px']
    X_px_val = val_df[px_cols_val]
    y_px_val = val_df['px']

    #Fitting linear regression on the training data, then predicting and scoring on the validation set:
    lm = LinearRegression()
    lm.fit(X_px_tr, y_px_tr)
    px_pred = lm.predict(X_px_val)
    px_r2 = lm.score(X_px_val, y_px_val)
    px_mae = mae(y_px_val, px_pred)
    print('Val Px R^2: {}'.format(px_r2))
    print('Val Px MAE: {} ft.'.format(px_mae))

    #Mapping the predicted px values onto the validation set:
    val_df['px_pred'] = px_pred

    return lm, val_df


def pz_linear_regression(train_df, val_df):
    '''
    Arguments: takes in a training and validation dataframe.
    Returns: a linear regression model of the z location of the pitch (pz), with the predicted values mapped onto the validation dataframe input.
    '''
    #Setting up the columns needed:
    pz_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'cumulative_ff_rate', 'cumulative_sl_rate', 'cumulative_ft_rate',
       'cumulative_ch_rate', 'cumulative_cu_rate', 'cumulative_si_rate',
       'cumulative_fc_rate', 'cumulative_kc_rate', 'cumulative_fs_rate',
       'cumulative_kn_rate', 'cumulative_ep_rate', 'cumulative_fo_rate',
       'cumulative_sc_rate', 'Last_Pitch_Type_Num', 'Pitch_Type_Num', 'px']
    pz_cols_val = pz_cols.copy()
    pz_cols_val.append('px_pred')
    pz_cols_val.remove('px')

    #Setting up the x and y inputs for linear regression:
    X_pz_tr = train_df[pz_cols]
    y_pz_tr = train_df['pz']
    X_pz_val = val_df[pz_cols_val]
    y_pz_val = val_df['pz']

    #Fitting linear regression on the training data, then predicting and scoring on the validation set:
    lm = LinearRegression()
    lm.fit(X_pz_tr, y_pz_tr)
    pz_pred = lm.predict(X_pz_val)
    pz_r2 = lm.score(X_pz_val, y_pz_val)
    pz_mae = mae(y_pz_val, pz_pred)
    print('Val Pz R^2: {}'.format(pz_r2))
    print('Val Pz MAE: {} ft.'.format(pz_mae))

    #Mapping the predicted px values onto the validation set:
    val_df['pz_pred'] = pz_pred

    return lm, val_df

def px_rf_regression(train_df, val_df):
    '''
    Arguments: takes in a training and validation dataframe.
    Returns: a Random Forest regression model of the x location of the pitch (px), with the predicted values mapped onto the validation dataframe input.
    '''
    #Setting up the columns needed:
    px_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'cumulative_ff_rate', 'cumulative_sl_rate', 'cumulative_ft_rate',
       'cumulative_ch_rate', 'cumulative_cu_rate', 'cumulative_si_rate',
       'cumulative_fc_rate', 'cumulative_kc_rate', 'cumulative_fs_rate',
       'cumulative_kn_rate', 'cumulative_ep_rate', 'cumulative_fo_rate',
       'cumulative_sc_rate', 'Last_Pitch_Type_Num', 'Pitch_Type_Num']
    px_cols_val = px_cols.copy()
    px_cols_val.append('pitch_pred')
    px_cols_val.remove('Pitch_Type_Num')

    #Setting up the x and y inputs for linear regression:
    X_px_tr = train_df[px_cols]
    y_px_tr = train_df['px']
    X_px_val = val_df[px_cols_val]
    y_px_val = val_df['px']

    #Fitting linear regression on the training data, then predicting and scoring on the validation set:
    rf = RandomForestRegressor(criterion='mae')
    rf.fit(X_px_tr, y_px_tr)
    px_pred = rf.predict(X_px_val)
    px_r2 = rf.score(X_px_val, y_px_val)
    px_mae = mae(y_px_val, px_pred)
    print('Val Px R^2: {}'.format(px_r2))
    print('Val Px MAE: {} ft.'.format(px_mae))

    #Mapping the predicted px values onto the validation set:
    val_df['px_pred'] = px_pred

    return rf, val_df


def pz_rf_regression(train_df, val_df):
    '''
    Arguments: takes in a training and validation dataframe.
    Returns: a Random Forest regression model of the z location of the pitch (pz), with the predicted values mapped onto the validation dataframe input.
    '''
    #Setting up the columns needed:
    pz_cols = ['Cluster','inning', 'top', 'on_1b', 'on_2b', 'on_3b', 'b_count', 's_count', 'outs', 'stand_R',
       'pitcher_run_diff','last_pitch_speed', 'last_pitch_px', 'last_pitch_pz','pitch_num','cumulative_pitches',
       'cumulative_ff_rate', 'cumulative_sl_rate', 'cumulative_ft_rate',
       'cumulative_ch_rate', 'cumulative_cu_rate', 'cumulative_si_rate',
       'cumulative_fc_rate', 'cumulative_kc_rate', 'cumulative_fs_rate',
       'cumulative_kn_rate', 'cumulative_ep_rate', 'cumulative_fo_rate',
       'cumulative_sc_rate', 'Last_Pitch_Type_Num', 'Pitch_Type_Num', 'px']
    pz_cols_val = pz_cols.copy()
    pz_cols_val.append('px_pred')
    pz_cols_val.remove('px')

    #Setting up the x and y inputs for linear regression:
    X_pz_tr = train_df[pz_cols]
    y_pz_tr = train_df['pz']
    X_pz_val = val_df[pz_cols_val]
    y_pz_val = val_df['pz']

    #Fitting linear regression on the training data, then predicting and scoring on the validation set:
    rf = RandomForestRegressor(criterion='mae')
    rf.fit(X_px_tr, y_px_tr)
    px_pred = rf.predict(X_px_val)
    px_r2 = rf.score(X_px_val, y_px_val)
    pz_mae = mae(y_pz_val, pz_pred)
    print('Val Pz R^2: {}'.format(pz_r2))
    print('Val Pz MAE: {} ft.'.format(pz_mae))

    #Mapping the predicted px values onto the validation set:
    val_df['pz_pred'] = pz_pred

    return rf, val_df

def pitch_prediction_modeling_pipeline(player_name, data, split_size = 0.2, class_method = 'RandomForest', reg_method = 'Linear'):
    '''
    Arguments: takes in a player name and a dataframe of MLB league-wide pitch data.  The default split size for train/validation by at bat is 0.20.
    Returns: Runs pitch classification modeling via Random Forest for pitch types, then pitch location prediction via linear regression.  
    Outputs a dataframe with the predicted values mapped onto the original validation dataframe.
    '''
    print('Pitch Modeling for {}'.format(player_name)) #printing the pitcher name, so it's clear to the user

    #First, setting up the dataframe and splitting the data into train and validation sets:
    training_pitches, val_pitches = dataframe_setup(player_name, data, split_size = 0.2)

    #Classification - Random Forest or XGBoost?
    if class_method == 'RandomForest':
        #Running random forest classification pitch type predictions:
        rf_model, val_df = random_forest_pitch_classification(training_pitches, val_pitches)
    
    elif class_method == 'XGBoost':
        xg_model, val_df = xgboost_pitch_classification(training_pitches, val_pitches)
    
    if reg_method == 'Linear'
        #Running linear regression predictions on the pitch x coordinate (px):
        px_model, val_df = px_linear_regression(training_pitches, val_df)
        
        #Running linear regression predictions on the pitch z coordinate (pz):
        pz_model, val_df = pz_linear_regression(training_pitches, val_df)
    
    elif reg_method == 'RandomForest':
        #Running Random Forest regression predictions on the pitch x coordinate (px):
        px_model, val_df = px_rf_regression(training_pitches, val_df)
        
        #Running Random Forest regression predictions on the pitch z coordinate (pz):
        pz_model, val_df = pz_rf_regression(training_pitches, val_df)
    #returning the validation dataframe with the predictions mapped out:
    return val_df

def multiple_pitcher_predictions(player_name_list, data, split_size = 0.2, class_method = 'RandomForest', reg_method = 'Linear'):
    '''
    Arguments: takes in a list of player names and a dataframe of MLB league-wide pitch data.  The default split size for train/validation by at bat is 0.20.
    Returns: Runs pitch classification modeling via Random Forest for pitch types, then pitch location prediction via linear regression.  
    Outputs a dataframe with the predicted values mapped onto the original validation dataframe.
    '''
    counter = 0
    for player in player_name_list:
        val_df = pitch_prediction_modeling_pipeline(player, data, split_size = split_size, class_method=class_method, reg_method=reg_method)
        #Adding in a check to initiate the output dataframe for the first loop through:
        if counter == 0:
            output_df = pd.DataFrame(columns=val_df.columns)
        output_df = pd.concat([output_df, val_df])
        counter += 1
        print('\n')
        print('\n')
    return output_df











