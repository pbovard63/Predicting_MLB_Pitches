'''
Functions to build out proportions of the pitch data for a pitcher's arsenal of pitches.
'''
import numpy as np
import pandas as pd

def pitch_rate_generator(df):
    '''
    Arguments: takes in a dataframe of pitch by pitch data.
    Returns: pitch counts by pitcher, with the rate the pitcher threw that pitch type as well.
    '''
    #First, getting the total number of pitches each pitcher in the input df threw:
    pitcher_total_pitches = df.groupby(['pitcher_full_name']).agg({'pitch_type': ['count']})
    
    #Next, building a dataframe with the pitches per type:
    pitcher_type_df = df.groupby(['pitcher_full_name','pitch_type']).agg({'pitch_type': ['count']})
    pitcher_type_df['Total_Pitches'] = 0
    
    #Looping through to add how many pitches each pitcher threw in total:
    for i, pitcher in enumerate(pitcher_type_df.index):
        pitcher_name = pitcher[0]
        total_pitches = pitcher_total_pitches.loc[pitcher_name, ('pitch_type', 'count')]
        pitcher_type_df.iloc[i, 1] = total_pitches
        
    #Adding in a pitch rate stat:
    pitcher_type_df['Pitch_Type_Rate'] = pitcher_type_df[('pitch_type', 'count')] / pitcher_type_df[('Total_Pitches', )]
    pitcher_type_df.reset_index(inplace=True)

    #Better Organizing the New DataFrame Columns:
    rate_df = pd.DataFrame(data=pitcher_type_df.values,columns=['Pitcher_Name', 'Pitch_Type', 'Pitch_Type_Count', 'Total_Pitch_Count', 'Pitch_Type_Rate'])
    
    #REturning the final dataframe
    return rate_df 

def pitch_arsenal_generator(df, threshold=0.05):
    '''
    Arguements: takes in a dataframe of pitch by pitch data, and a minimum threshold for pitch rate, for a pitch to be included in the arsenal. Default threshold is 0.05.
    Returns: a dataframe with pitcher names and their respective thresholds (w/rates).
    '''
    #First, running pitch_rate_generator with the inputted dataframe:
    rate_df =  pitch_rate_generator(df)

    #Creating a column to code a pitch as in or out of the arsenal, depending on the threshold argument:
    rate_df['In_Arsenal'] = 0
    for i, rate in enumerate(rate_df.Pitch_Type_Rate):
        if rate >= threshold:
            rate_df.iloc[i, 5] = 1
        else:
            rate_df.iloc[i, 5] = 0

    #Pulling pitcher names from the rate_df:
    pitcher_names = rate_df.Pitcher_Name.value_counts().index

    #Building the Arsenal by pitcher:
    arsenal_list = []
    for pitcher in pitcher_names:
        pitcher_df = rate_df[rate_df.Pitcher_Name == pitcher]
        pitcher_arsenal = {}
        for i, pitch_type in enumerate(pitcher_df.Pitch_Type):
            if pitcher_df.In_Arsenal.iloc[i] == 1:
                pitcher_arsenal[pitch_type] = pitcher_df.Pitch_Type_Rate.iloc[i]
        arsenal_list.append(pitcher_arsenal)

    #Building a new dataframe with pitcher names and arsenals:
    data = {'Pitcher_Name':pitcher_names, 'Pitch_Arsenal':arsenal_list}
    pitcher_arsenal_df = pd.DataFrame(data=data)
    return pitcher_arsenal_df

def pitcher_pitch_rate_gen(df, threshold=0.05):
    '''
    Arguements: takes in a dataframe of pitch by pitch data, and a minimum threshold for pitch rate, for a pitch to be included in the arsenal. Default threshold is 0.05.
    Returns: a dataframe with pitcher names as rows and columns of pitch type, with values being frequency thrown (if above the given threshold, otherwise it will default to 0).
    '''
    #First, running the previous functions:
    rate_df =  pitch_rate_generator(df)
    arsenal_df = pitch_arsenal_generator(df, threshold)

    #Creating a new dataframe, with pitchers as rows and pitch type as columns:
    data = {'Pitcher_Name':arsenal_df.Pitcher_Name.values}
    new_df = pd.DataFrame(data=data)

    #Setting up column names:
    for pitch_type in rate_df.Pitch_Type.value_counts().index:
        new_df[pitch_type] = 0
    new_df['Pitches_In_Arsenal'] = 0

    #Filling in pitch rates:
    for i, arsenal in enumerate(arsenal_df.Pitch_Arsenal):
        new_df['Pitches_In_Arsenal'].iloc[i] = len(arsenal)
        for pitch, rate in arsenal.items():
            new_df[pitch].iloc[i] = rate

    return new_df
        