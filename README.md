# Predicting MLB Pitch Types :baseball:
## By: Patrick Bovard
### *Final Passion Project for Metis Winter 2021*

### Project Intro Information 
**Project Description:** This project will utilizing MLB Pitch by Pitch Data to predict what pitch type, and in what location, a pitcher will throw to a batter, given a set of features.

**Data Sources:** Data from the following sources has been utilized in this project:  
- Paul Schale's [MLB Pitch Data Kaggle Dataset](https://www.kaggle.com/pschale/mlb-pitch-data-20152018?select=games.csv).  Includes MLB Pitch data from 2015-2019.
- Baseball Savant's [website](https://baseballsavant.mlb.com/).  This site was used to look at pitch trends by pitchers over time, as seen in my final presentation.  
- [FanGraphs Custom Leaderboards](https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=5&season=2019&month=0&season1=2015&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=2015-01-01&enddate=2019-12-31).  Data from here was used to cluster hitters into different "types" of hitters.

**Main Tools Utilized:** 
- Data Analysis and Model Building: Python, Pandas, Numpy, Scikit-Learn, SQL-Alchemy, PostgreSQL  
- Pitch Location: Linear Regression Modeling
- Pitch Type Classification: XGBoost Classification Modeling  
- Batter Clustering: K-Means Clustering

**Possible Impacts:** Possible impacts of this project are as a gameplanning tool, that a batter could use to prepare.  While it couldn't be utilized mid-game, it could be used as a preparation tool while getting ready to face a certain pitcher, to create customized scouting reports for hitters based on situations they may see.

### Navigating the repo:
Below has a main overview of the files in my repo, and what each folder contains.  For more detailed information, check out the Table_of_Contents.ipynb notebook in the main area of the repo.  *Note: several code files refer to a "Data" folder.  This was added to .gitignore due to the large file size and does not appear in this GitHub repo.*
- **MLB_Pitch_Data_Setup_SQL Folder:** This folder houses notebooks and code used to create SQL database of data from the [Kaggle dataset](https://www.kaggle.com/pschale/mlb-pitch-data-20152018?select=games.csv) mentioned above.   
- **EDA-SQL Folder:** This folder contains code on initial EDA of the data, including SQL and Pandas code to structure the dataset in preparation for machine learning.  
- **Clustering Folder:** This folder houses code on using K-Means clustering on batters, in order to dimensionally reduce the features with batters to feed into the machine learning algorithms.
- **Pitch_Classification Folder:** This folder contains code on building the following machine learning models: 
  - Classification: Determining pitch type
  - Linear Regression: Predicting pitch location (x and y coordinates)  
- **Pitch_Classification_Folder:** This folder contains my code on building out the machine learning models.  This is divided into subfolders:
  - Modeling_Preparation, to prepare the dataframe/features for modeling
  - Individual_Pitcher_Runs, to establish the General Pipeline for modeling
  - Pipeline_Building, to develop a pipeline for chaining together the regression and classification algorithms, and to incorporate various pitchers
  - Final_Modeling: contains notebooks and functions utilized in the final model
- **Feature_Engineering_Additional Folder:** This folder contains code on running additional feature engineering to utilize in my modeling process.  
- **Final_Presentation Folder:** This folder contains my final presentation slides, in PDF and Powerpoint file formats.  
- **Streamlit_App Folder:** This folder contains the script for running the Streamlit App to show my final model and results. *Note: this is currently under construction.*

