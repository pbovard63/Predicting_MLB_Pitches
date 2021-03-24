'''
This file will contain the script for running my project's streamlit web app.
'''
#Importing packages:
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Introduction part:
st.write('''
# MLB Pitch Predictor
### By: Patrick Bovard
*Welcome! This app was made as part of my final project at Metis, to predict MLB pitches.  If you'd like to check out the source code, my [GitHub repo for this project is here](https://github.com/pbovard63/metis_final_project)!*

**Project Description**:
The goal of this project was to be able to predict what pitch a batter may see.  With this app, you'll be able to get a scouting overview on the selected pitcher, as well as predictions on where a pitch would be.
'''
)

#Bringing in data:
with open('./Data/actual_final_df.pickle','rb') as read_file:
    pitch_df = pickle.load(read_file)
pitcher_list = pitch_df.pitcher_full_name.value_counts().head(10).index

st.write('''
# Let's get started!  Which pitcher do you want to learn more about?
'''
)
user_input = st.selectbox('Select a pitcher name:',
pitcher_list)

if user_input:
    st.write(f'# Scouting Report for {user_input}:')
    st.write(f'Here is a quick scouting breakdown on {user_input}, based on the pitches they throw and where their location is.')

    st.write('### Pitch Type Breakdown:')
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
    ax1.set_yticklabels("")

    show_graph = st.checkbox('Show Graph', value=True)
    if show_graph:
        st.pyplot(fig1)

    #Setting up plots for pitch location:
    st.write('### Pitch Locations')
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
    ax2.set_xticklabels("")
    ax2.set_yticklabels("")
    strikezone_layout()
    st.pyplot(fig2)

    #Adding in the necessary inputs to run predictions:
    st.write('''
    # Pitch Predictions for {}
    Before I can give you a prediction, I need to know some information.  Below, enter in information requested for the scenario, and you will be provided with a prediction for the pitch type and location that {} will throw.
    '''.format(user_input, user_input)
    )
    #Selecting a hitter, based on the hitting clusters behind my model:
    st.write('''
    ## Preliminary Information
    ### Batter:
    Choose your batter from the list below:
    ''')
    batter_list = ['Giancarlo Stanton (R)', 'Eric Thames (L)', 'Dee Strange-Gordon (L)','Billy Hamilton (S)', 'Ozzie Albies (S)', 'James McCann (R)']
    #Cluster 0: Thames (L), Stanton (R), Cluster 3: Gordon (L), Billy Hamilton (S), Cluster 1: James McCann (R), Ozzie Albies (S)
    batter_choice = st.selectbox('Select a batter name from the below list:', batter_list)
    #After the batter is selected, assigning the batter cluster to use in predictions:
    if batter_choice == 'Giancarlo Stanton' or batter_choice == 'Eric Thames':
        batter_cluster = 0
    elif batter_choice == 'Dee Strange-Gordon' or batter_choice == 'Billy Hamilton':
        batter_cluster = 3
    elif batter_choice == 'James McCann' or batter_choice == 'Ozzie Albies':
        batter_cluster = 1
    
    #Assigning hitter handedness for 'stand_R' feature:
    if '(R)' in batter_choice:
        stand_R = 1
    else:
        stand_R = 0

    st.write('### Game Situation Information')
    st.write('{} is up to bat.  Enter in game situation information below.'.format(batter_choice))
    inning = st.slider('Inning Number', min_value=1, max_value=9, step=1)
    inning_side = st.selectbox('Top or Bottom of the Inning?', ['Top', 'Bottom'])
    if inning_side == 'Top':
        top = 1
    else:
        top = 0

    #Setting the count:
    outs = st.slider('Number of Outs', min_value=0, max_value=2, step=1)
    strikes = st.slider('Number of Strikes', min_value=0, max_value=2, step=1)
    balls = st.slider('Number of Balls', min_value=0, max_value=3, step=1)
    
    #Runners on base:
    base_states = ['Bases Empty', 'Runner on First', 'Runner on Second', 'Runner on Third', 'Runners on First and Second', 'Runners on First and Third', 'Runners on Second and Third', 'Bases Loaded']
    runners = st.selectbox('Are there runners on base?', base_states)
    #once runners on base are selected, setting the on_1b, on_2b, and on_3b parameters, setting all to 0 by default:
    on_1b = 0
    on_2b = 0
    on_3b = 0
    if runners == 'Runner on First' or runners == 'Runners on First and Second' or runners == 'Runners on First and Third' or runners == 'Bases Loaded':
        on_1b = 1
    if runners == 'Runner on Second' or runners == 'Runners on Second and Third' or runners == 'Runners on First and Second' or runners == 'Bases Loaded':
        on_2b = 1
    if runners == 'Runner on Third' or runners == 'Runners on First and Third' or runners == 'Runners on Second and Third' or runners == 'Bases Loaded':
        on_3b = 1

    #Moving into at-bat specific information:
    st.write('### At-Bat Information:')
    pitch_nums = [0,1,2,3,4,5,6,7,8,9,10]
    ab_pitch_num = st.selectbox('How many pitches have you seen this at bat?', pitch_nums)
    model_pitch_num = ab_pitch_num + 1 #Modeling pitch num is the pitch number of the at bat, so +1 to how many pitches have been already seen

    #If this isn't the first pitch of the at bat, askign for the last pitch info
    if ab_pitch_num > 0:
        st.write('Now, some information on the last pitch thrown, please:')
        last_pitch_type = st.selectbox('Select the last pitch type:', pitch_types)   
        last_pitch_type_num = 3 #PLACE HOLDER   
        pitch_speed_max = pitcher_data[pitcher_data.pitch_type==last_pitch_type].start_speed.max()
        pitch_speed_min = pitcher_data[pitcher_data.pitch_type==last_pitch_type].start_speed.min()
        last_pitch_speed = st.slider('Select the last pitch speed:', min_value = int(pitch_speed_min), max_value = int(pitch_speed_max), step=2)

    #need to define last pitch info for pitch number = 0

    st.write('''
    ## Pitch Predictions:
    Based on the provided information, here are the likely pitch types:
    '''
    )
    #Bringing in the models, and specifying the ones for the pitcher:
    with open('./Data/final_models_dict.pickle','rb') as read_file:
        model_dict = pickle.load(read_file)
    #Extracting the whole model set, and the individual ones:
    pitcher_models = model_dict[user_input]
    xgb_model = pitcher_models[0][0]
    px_model = pitcher_models[0][1]
    pz_model = pitcher_models[0][2]

    #With the model imported and loaded, will need to predict on the given data:
    #First, need to pull the most recent row of data on that pitcher, for their cumulative rates and last 5, etc:
    last_row = pitcher_data.tail(1)
    cumulative_pitches = last_row.loc[:,'cumulative_pitches']
    cumulative_ff_rate = last_row.loc[:,'cumulative_ff_rate']
    cumulative_sl_rate = last_row.loc[:,'cumulative_sl_rate']
    cumulative_ft_rate = last_row.loc[:,'cumulative_ft_rate']
    cumulative_ch_rate = last_row.loc[:,'cumulative_ch_rate']
    cumulative_cu_rate = last_row.loc[:,'cumulative_cu_rate']
    cumulative_si_rate = last_row.loc[:,'cumulative_si_rate']
    cumulative_fc_rate = last_row.loc[:,'cumulative_fc_rate']
    cumulative_kc_rate = last_row.loc[:,'cumulative_kc_rate']
    cumulative_fs_rate = last_row.loc[:,'cumulative_fs_rate']
    cumulative_kn_rate = last_row.loc[:,'cumulative_kn_rate']
    cumulative_ep_rate = last_row.loc[:,'cumulative_ep_rate']
    cumulative_fo_rate = last_row.loc[:,'cumulative_fo_rate']
    cumulative_sc_rate = last_row.loc[:,'cumulative_sc_rate']
    last_5_ff = last_row.loc[:,'last_5_ff']
    last_5_sl = last_row.loc[:,'last_5_sl']
    last_5_ft = last_row.loc[:,'last_5_ft']
    last_5_ch  = last_row.loc[:,'last_5_ch']
    last_5_cu = last_row.loc[:,'last_5_cu']
    last_5_si = last_row.loc[:,'last_5_si']
    last_5_fc = last_row.loc[:,'last_5_fc']
    last_5_kc = last_row.loc[:,'last_5_kc']
    last_5_fs = last_row.loc[:,'last_5_fs']
    last_5_kn = last_row.loc[:,'last_5_kn']
    last_5_ep = last_row.loc[:,'last_5_ep']
    last_5_fo = last_row.loc[:,'last_5_fo']
    last_5_sc = last_row.loc[:,'last_5_sc']

    #Below values are placeholders to use in testing:
    pitcher_run_diff = 1
    last_pitch_px = 0
    last_pitch_pz = 1
    batter_cluster = 0

    #XGBoost: needed features are: (need stand_R, pitcher_run_diff, last_pitch_px, last_pitch_pz, )
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
    
    xgb_prediction_array = np.array([batter_cluster, inning, top, on_1b, on_2b, on_3b, balls, strikes, outs, stand_R, pitcher_run_diff, 
    last_pitch_speed, last_pitch_px, last_pitch_pz, model_pitch_num, 
    cumulative_pitches, cumulative_ff_rate, cumulative_sl_rate, cumulative_ft_rate, cumulative_ch_rate, cumulative_cu_rate, 
    cumulative_si_rate, cumulative_fc_rate, cumulative_kc_rate, cumulative_fs_rate, cumulative_kn_rate, cumulative_ep_rate, 
    cumulative_fo_rate,  cumulative_sc_rate, last_5_ff, last_5_sl, last_5_ft, last_5_ch, last_5_cu, last_5_si, last_5_fc, last_5_kc, 
    last_5_fs, last_5_kn, last_5_ep, last_5_fo, last_5_sc])
    
    xg_df = pd.DataFrame(data=xgb_prediction_array)
    X = xg_df
    print(X.dtypes)

    pitch_type_pred = xgb_model.predict(X)[0]
    #Need to map the pitch type from a number to a pitch type:
    pitch_type_pred_word = pitch_types[pitch_type_pred]

    pitch_type_pred_probs = xgb_model.predict_proba(X)

    #Lin. Reg: Px (need stand_R, pitcher_run_diff, last_pitch_px, last_pitch_pz, )
    #px_prediction_array = np.array[cluster, inning, top, on_1b. on_2b, on_3b, balls, strikes, outs, stand_R, pitcher_run_diff, 
    #last_pitch_speed, last_pitch_px, last_pitch_pz, model_pitch_num, cumulative_pitches

    #px_pred = px_model.predict(px_prediction_array)

    #Lin. Reg: Pz (need stand_R, pitcher_run_diff, last_pitch_px, last_pitch_pz, )
    #px_prediction_array = np.array[cluster, inning, top, on_1b. on_2b, on_3b, balls, strikes, outs, stand_R, pitcher_run_diff, 
    #last_pitch_speed, last_pitch_px, last_pitch_pz, model_pitch_num, cumulative_pitches, px_pred]

    #pz_pred = px_model.predict(px_prediction_array)

    #Writing out the prediction for the user:
    st.write('''
    # Based on the situation, the predicted pitch type is:
    **{}**
    '''.format(pitch_type_pred_word)
    )



        