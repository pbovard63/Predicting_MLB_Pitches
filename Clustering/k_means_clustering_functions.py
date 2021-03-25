'''
This file contains functions for running K-means clustering on a dataframe of hitting statistics, to group similar hitters together in a dimensionality reduction of their full stats.
'''
#Importing needed packages:
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle
from sqlalchemy import create_engine
import pandas as pd
from importlib import reload
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

def inertia_plotter(X, max_clusters=10):
    '''
    Arguments: takes in a max number of clusters to run K-Means clustering on the inputted dataset x.  Default max_clusters is 10.
    Returns: a plot of inertia with the number of clusters.
    '''
    inertia = []

    #Making list of clusters:
    max_val = max_clusters + 1
    list_num_clusters = list(range(1,max_val))
    
    #Running K-Means on the dataset:
    for num_clusters in list_num_clusters:
        km = KMeans(n_clusters=num_clusters)
        km.fit(X)
        inertia.append(km.inertia_)
    
    #Plotting:
    plt.plot(list_num_clusters,inertia)
    plt.scatter(list_num_clusters,inertia)
    plt.title('Inertia against Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia');

def cluster_assigner(df, X_cols, num_clusters, standard_scaler=True):
    '''
    Arguments: takes in a dataset df with columns X_cols to run K-Means clustering on.  Also, takes in a standard_scaler argument, default set to True.  If True, will standard scale all columns in X.
    Also takes in num_clusters, which is the number of clusters to use in K-means clustering.
    Returns: a dataframe with clusters assigned after running K-means clustering.
    '''
    #Pulling the needed data from df:
    X = df[X_cols]

    #Standard scaling, if selected:
    if standard_scaler:
        std = StandardScaler()
        X_scaled = std.fit_transform(X)
    
    #Running K-Means clustering:
    km = KMeans(n_clusters=num_clusters,random_state=10,n_init=1) # n_init, number of times the K-mean algorithm will run
    km.fit(X_scaled)

    #Assigning Clusters:
    assignments = km.labels_
    output_df = df.copy()
    output_df['Cluster'] = assignments
    return output_df