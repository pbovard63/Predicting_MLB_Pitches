'''
This file will contain the script for running my project's streamlit web app.
'''
#Importing packages:
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Introduction part:
st.write('''
# MLB Pitch Predictor
*This app was made as part of my final project at Metis, to predict MLB pitches.  Take a look a look around here, or my [GitHub repo](https://github.com/pbovard63/metis_final_project)!*

**Project Description**:
The goal of this project was to be able to predict what pitch a batter may see
'''
)

#Bringing in data:
with open('./Data/train_df_clusters.pickle','rb') as read_file:
    pitch_df = pickle.load(read_file)
pitcher_list = pitch_df.pitcher_full_name.value_counts().head(10).index

st.write('''
# Let's get started!  Which pitcher do you want to learn more about?
'''
)
user_input = st.selectbox('Select a pitcher name:',
pitcher_list)

if user_input:
    st.write(f'Here is a quick scouting breakdown on {user_input}:')

    #Setting up the dataframe, for that pitcher:
    pitcher_data = pitch_df[pitch_df.pitcher_full_name == user_input]
    st.write(f'Here are the pitches {user_input} throws:')
    pitch_types = pitcher_data.pitch_type.value_counts().index
    pitch_counts = pitcher_data.pitch_type.value_counts()

    pitch_rates = [value/pitcher_data.shape[0] for value in pitch_counts]
    
    fig1, ax1 = plt.subplots()
    ax1.bar(pitch_types, pitch_rates)
    ax1.set_title(f'Pitch Types and Rate for {user_input}')
    ax1.set_xlabel(f'Pitch Types thrown by {user_input}')
    ax1.set_ylabel('Percentage of Times Thrown')

    show_graph = st.checkbox('Show Graph', value=True)
    if show_graph:
        st.pyplot(fig1)

    #Setting up plots for pitch location:
    st.write(f'For each pitch type, here is the spread of where {user_input} throws them:')
    
    #Default Strikezone imagine layout:
    def strikezone_layout():
        plt.xlim(-4, 4)
        plt.ylim(0,5)
        plt.hlines(y=1.57, xmin=-0.71, xmax=0.71)
        plt.hlines(y=3.42, xmin=-0.71, xmax=0.71)
        plt.vlines(x=-0.71, ymin=1.57, ymax=3.42)
        plt.vlines(x=0.71, ymin=1.57, ymax=3.42)
        plt.show();
    
    pitch_input = st.selectbox('Select a pitcher type:', pitch_types)
    fig2, ax2 = plt.subplots()
    ax2.scatter(pitcher_data['px'][pitcher_data.pitch_type == pitch_input], pitcher_data['pz'][pitcher_data.pitch_type == pitch_input], alpha = 0.05)
    ax2.set_title(f'{user_input} Actual Pitch Locations - 2015-2018, {pitch_input}')
    strikezone_layout()
    st.pyplot(fig2)
        
