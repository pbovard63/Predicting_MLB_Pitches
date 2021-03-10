'''
This file contains functions to be used in running linear regression to predict the pitch coordinates.
'''
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

def mae(y_true, y_pred):
    '''
    Arguments: true target (y_true) and predicted target (y_pred) values for a dataset.
    Returns: the mean average error between the actual and predicted values.
    '''
    mae_output = np.mean(np.abs(y_pred - y_true))
    return mae_output

def split_and_train_val_simple_lr(X, y):
    '''
    Argument: takes in a set of features X and a target variable y.
    Returns: Performs simple linear regression and returns the feature coefficeints and validation R^2.
    '''
    #First, split into train/val and test sets of data:
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.2, random_state=5)
    
    #Simple Linear Regression (no CV):
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=.25, random_state=12)
    
    #Standard Scaling of features:
    std = StandardScaler()
    std.fit(X_train.values)
    X_train_scaled = std.transform(X_train.values)
    X_val_scaled = std.transform(X_val.values)
    
    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train)
    print('Simple Linear Regression Results:')
    print(f'Simple Linear Regression val R^2: {lm.score(X_val_scaled, y_val):.3f}')
    print(list(zip(X.columns, lm.coef_)))

def split_and_train_val_simple_lr_w_cv(X, y):
    '''
    Argument: takes in a set of features X and a target variable y.
    Returns: Performs simple linear regression w/ K Fold cross validation and returns the feature coefficeints and validation R^2.
    '''
    #Split data from train_val_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.2, random_state=5)
    
    #Standard scaling of features:
    std = StandardScaler()
    std.fit(X_train_val.values)
    X_train_val_scaled = std.transform(X_train_val.values)
    
    #Converting the cross validation sets to numpy arrays:
    X_cv, y_cv = np.array(X_train_val_scaled), np.array(y_train_val)
    kf = KFold(n_splits=5, shuffle=True, random_state = 12)
    cv_lm_r2s = []
    
    #Creating different CV sets:
    for train_ind, val_ind in kf.split(X_cv,y_cv):
    
        X_train, y_train = X_cv[train_ind], y_cv[train_ind]
        X_val, y_val = X_cv[val_ind], y_cv[val_ind] 
    
        #simple linear regression
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        cv_lm_r2s.append(lm.score(X_val, y_val))

    print('Simple Linear Regression w/ KFOLD CV Results:')
    print('Simple regression scores: ', cv_lm_r2s, '\n')
    print(f'Simple mean cv r^2: {np.mean(cv_lm_r2s):.3f} +- {np.std(cv_lm_r2s):.3f}')
    
def split_and_train_val_lasso(X, y):
    '''
    Argument: takes in a set of features X and a target variable y.
    Returns: Performs lasso linear regression w/ cross validation and returns the feature coefficeints and validation R^2.
    '''
    #Split data from train_val_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.2, random_state=5)
    
    #Standard scaling features:
    std = StandardScaler()
    std.fit(X_train_val.values)
    X_train_val_scaled = std.transform(X_train_val.values)
    
    #Setting up Lasso Model:
    lasso_model = LassoCV(cv= 5)
    lasso_model.fit(X_train_val_scaled, y_train_val)
    
    #Predicting and scoring model:
    train_val_pred = lasso_model.predict(X_train_val_scaled)
    lasso_mae = mae(y_train_val, train_val_pred)
    lasso_r2 = r2_score(y_train_val, train_val_pred)
    print('Lasso Linear Regression w/ CV Results:')
    print('Lasso R^2: {}'.format(lasso_r2))
    print('Lasso mae: {}'.format(lasso_mae))
    print('Lasso Coefficients: {}'.format(list(zip(X.columns, lasso_model.coef_))))

def split_and_train_val_ridge(X, y):
    '''
    Argument: takes in a set of features X and a target variable y.
    Returns: Performs ridge linear regression w/ cross validation and returns the feature coefficeints and validation R^2.
    '''
    #Split data from train_val_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.2, random_state=5)
    
    #Standard Scaling Features:
    std = StandardScaler()
    std.fit(X_train_val.values)
    X_train_val_scaled = std.transform(X_train_val.values)
    
    #Setting up ridge model:
    ridge_model = RidgeCV(cv= 5)
    ridge_model.fit(X_train_val_scaled, y_train_val)
    
    #Making prediction and scoring model:
    train_val_pred = ridge_model.predict(X_train_val_scaled)
    ridge_mae = mae(y_train_val, train_val_pred)
    ridge_r2 = r2_score(y_train_val, train_val_pred)
    print('Ridge Linear Regression w/ CV Results:')
    print('Ridge R^2: {}'.format(ridge_r2))
    print('Ridge mae: {}'.format(ridge_mae))
    print('Ridge Coefficients: {}'.format(list(zip(X.columns, ridge_model.coef_))))

def split_and_train_val_EN(X, y):
    '''
    Argument: takes in a set of features X and a target variable y.
    Returns: Performs ElasticNet linear regression w/ cross validation and returns the feature coefficeints and validation R^2.
    '''
    #Split data from train_val_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.2, random_state=5)
    
    #Standard Scaling features:
    std = StandardScaler()
    std.fit(X_train_val.values)
    X_train_val_scaled = std.transform(X_train_val.values)
    
    #Setting up ElasticNet Model:
    ev_model = ElasticNetCV(cv= 5)
    ev_model.fit(X_train_val_scaled, y_train_val)
    
    #Making Prediction and scoring model:
    train_val_pred = ev_model.predict(X_train_val_scaled)
    ev_mae = mae(y_train_val, train_val_pred)
    ev_r2 = r2_score(y_train_val, train_val_pred)
    print('ElasticNet Linear Regression w/ CV Results:')
    print('ElasticNet R^2: {}'.format(ev_r2))
    print('ElasticNet mae: {}'.format(ev_mae))
    print('ElasticNet Coefficients: {}'.format(list(zip(X.columns, ev_model.coef_))))

def validation_comparer(X,y):
    '''
    Argument: takes in a set of features X and a target variable y.
    Returns: results of linear regression for simple LR, simple LR w/ KFold cross validation, and regularization via Ridge, Lasso, and ElasticNet.
    '''
    
    #Simply printing results for each model for easy comparison:
    
    #Simple Linear Regression:
    split_and_train_val_simple_lr(X, y)
    print('\n')
    
    #Simple Linear Regression, w// K Fold Cross validation:
    split_and_train_val_simple_lr_w_cv(X, y)
    print('\n')
    
    #Lasso Regularization:
    split_and_train_val_lasso(X, y)
    print('\n')
    
    #Ridge Regularization:
    split_and_train_val_ridge(X, y)
    print('\n')
    
    #ElasticNet Regularization:
    split_and_train_val_EN(X, y)
