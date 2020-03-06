import pymssql
import pandas as pd
import pyodbc
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from random import random
import datetime
from sql.sql_statements import select_matches_with_targets, match_aggregated_stats, last_matches_home_query, last_matches_away_query, last_direct_home_query
#DB Connection 
conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                     "Server=Martin-PC\SQLEXPRESS;"
                     "Database=FootballData;"
                     "Trusted_Connection=yes;")

# Excute Query here
df_matches = pd.read_sql(select_matches_with_targets,conn)
df_matches.drop_duplicates(['MatchId'], inplace=True)
df_matches['Date']=pd.to_datetime(df_matches['Date'])
#df_matches['day'] = df_matches['Date'].dt.day
#df_matches['month'] = df_matches['Date'].dt.month
#df_matches['year'] = df_matches['Date'].dt.year
#df_matches['dayofweek'] = df_matches['Date'].dt.dayofweek
#df_matches['daysago'] = (datetime.datetime.today() - df_matches['Date']).dt.days

print(df_matches.shape)
print(df_matches.head(5))

le = preprocessing.LabelEncoder()

def transform_columns(df_matches):
    for column in df_matches.columns:
        if df_matches[column].dtype != np.number:
            print(column + ' not a number')
            df_matches[column] = le.fit_transform(df_matches[column].astype(str)).astype('int32')
            print(df_matches[column])
            #print(set(s3_train_data_filtered[column]))
        else:
            print(column + " a number")
            df_matches[column] = df_matches[column].astype(np.float32)


# Get last x matches of home and away team
def get_match_features(match):
    #print(match)
    ''' Create match features for a given match. '''
    print("{} {}".format(match.Date, match.match_result))

    df_previous_matches_home = pd.read_sql(last_matches_home_query(match.HomeTeamId, match.Date),conn)
    df_previous_matches_home.drop(['MatchId','HomeTeamFullName','AwayTeamFullName','HomeTeamId','AwayTeamId','match_result','date'],axis =1,inplace=True)
    print(df_previous_matches_home.shape)

    df_previous_matches_home_sum=process_matches_average(df_previous_matches_home)

    df_previous_matches_away = pd.read_sql(last_matches_away_query(match.AwayTeamId, match.Date),conn)
    print(df_previous_matches_away.shape)

    df_previous_matches_away.drop(['MatchId','HomeTeamFullName','AwayTeamFullName','HomeTeamId','AwayTeamId','match_result','date'],axis =1,inplace=True)
    df_previous_matches_away_sum=process_matches_average(df_previous_matches_away)
    df_previous_matches_direct_home = pd.read_sql(last_direct_home_query(match.HomeTeamId, match.AwayTeamId, match.Date) ,conn)

    df_previous_matches_direct_home.drop(['MatchId','HomeTeamFullName','AwayTeamFullName','HomeTeamId','AwayTeamId','match_result','date'],axis =1,inplace=True)
    df_previous_matches_direct_sum=process_matches_average(df_previous_matches_direct_home)
    print(df_previous_matches_direct_home.shape)
#         match_possession_for = df_possessions[(df_possessions['MatchId'] == match_id)]['HomePossession']
#         result.loc[0, 'team_possession'] = np.mean(match_possession_for)
    data=[
        match.match_result,
        match.Date,
        match.HomeTeamFullName,
        match.AwayTeamFullname,       
        # match.day,
        # match.month,
        # match.year,
        # match.dayofweek,
        # match.daysago,
        df_previous_matches_home_sum, 
        df_previous_matches_away_sum,
        df_previous_matches_direct_sum,    
        ]
    
    data=np.hstack(data)
    home_away=pd.DataFrame(data)
    home_away=home_away.transpose()
    home_away.columns=np.hstack([
        'FTR',
        'Date',
        'HomeTeam',
        'AwayTeam',
        # 'day',
        # 'month',
        # 'year',
        # 'dayofweek',
        # 'daysago',
        'home_'+ df_previous_matches_home.columns,
        'away_'+ df_previous_matches_away.columns,
        'direct_'+ df_previous_matches_direct_home.columns,
    ])

    return home_away.iloc[0]

def process_matches_average(matches):
  
    #matches["home_team_possession_avg"] = matches["home_team_possession_avg"].mean()
    return matches.mean(axis = 0, skipna = True)

match_features = df_matches.apply(lambda x: get_match_features(x), axis = 1)
print(match_features.head())
match_features.to_csv('output/features1.csv',index=False)