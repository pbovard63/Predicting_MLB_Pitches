# Predicting MLB Pitch Types :baseball:
## By: Patrick Bovard
### *Final Passion Project for Metis Winter 2021*

### Project Intro Information 
**Project Description:** Utilizing MLB Pitch by Pitch Data to predict what pitch type, and in what location, a pitcher will throw to a batter. 

**Data Sources:** Data from the following sources has been utilized in this project:  
- Paul Schale's [MLB Pitch Data Kaggle Dataset](https://www.kaggle.com/pschale/mlb-pitch-data-20152018?select=games.csv).  Includes MLB Pitch data from 2015-2019.
- Baseball Savant's [Statcast search tool](https://baseballsavant.mlb.com/statcast_search).  Data was scraped from this site using Selenium and BeautifulSoup for 2020 pitch data.  
- [FanGraphs Custom Leaderboards](https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=5&season=2019&month=0&season1=2015&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=2015-01-01&enddate=2019-12-31).  Data from here was used to cluster hitters into different "types" of hitters.

**Tools Utilized:** 
- Webscraping: BeautifulSoup, Selenium
- Data Analysis and Model Building: Python, Pandas, Numpy, Scikit-Learn, SQL-Alchemy, PostgreSQL  
- Pitch Location: Linear Regression Modeling (Standard and Regularization via LassoCV, RidgeCV, and ElasticNetCV)
- Pitch Type Classification: Random Forest Classification  
- Batter Clustering: K-Means Clustering

**Possible Impacts:** Possible impacts of this project are as a gameplanning tool, that a batter could use to prepare.  While it couldn't be utilized mid-game, it could be used as a preparation tool while getting ready to face a certain pitcher.

### :construction: PROJECT IN PROGRESS :construction:  
*Estimated Completion Date: March 26, 2021*

### Navigating the repo:
- **Data Folder:** houses several of the data files used in this project, mainly in .csv or .pickle format.  *Note: several data files used have been added to .gitignore and may not appear in the online repo due to size constraints.*    
- **MLB_Pitch_Data_Setup_SQL:** houses notebooks and code used to create SQL database of data from the [Kaggle dataset](https://www.kaggle.com/pschale/mlb-pitch-data-20152018?select=games.csv) mentioned above.  
  - Initial_Kaggle_Dataset_Construction.ipynb: code on combining the .csv files into combined files for pitch, at bat, and games.
  - kaggle_dataset_sql_construction.ipynb: code on adding the combined .csv files into a PostgreSQL database, mlb_pitches.  
- **Other_Data_Sources:** code on gathering data from other sites, including Baseball Savant.  
- **EDA-SQL:** code on initial EDA of the data, including SQL and Pandas code to structure the dataset in preparation for machine learning.  
- **Pitch_Classification Folder:** code on building the following machine learning models:  
  - Classification: Determining pitch type
  - Linear Regression: Predicting pitch location (x and z coordinates)  
- **Clustering Folder:** code on clustering batters, in order to dimensionally reduce the features with batters to feed into the machine learning algorithms.
