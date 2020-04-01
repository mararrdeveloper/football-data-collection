import pymssql
import pandas as pd
import pyodbc
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from random import random
import datetime
from sql.sql_statements import select_matches_with_targets, match_aggregated_stats, last_matches_home_query, last_matches_away_query, last_direct_home_query


def drop_columns(matches):
    matches.sort_values('date', inplace=True, ascending=False)
    matches.drop(['MatchId','HomeTeamFullName','AwayTeamFullName','HomeTeamId','AwayTeamId','match_result','date'],axis =1,inplace=True)

def process_matches_average(matches):
    
    #matches["home_team_possession_avg"] = matches["home_team_possession_avg"].mean()
    return [matches[:1].mean(axis = 0, skipna = True), 
        matches[:2].mean(axis = 0, skipna = True),
        matches[:3].mean(axis = 0, skipna = True),
        matches[:4].mean(axis = 0, skipna = True),
        matches[:5].mean(axis = 0, skipna = True),
        matches[:6].mean(axis = 0, skipna = True)]


''' Create match features for a given match. '''
def get_match_features(match):
    df_previous_matches_home = pd.read_sql(last_matches_home_query(match.HomeTeamId, match.Date),conn)
    #print(df_previous_matches_home['date'])
    drop_columns(df_previous_matches_home)
    df_previous_matches_home_agg=process_matches_average(df_previous_matches_home)

    df_previous_matches_away = pd.read_sql(last_matches_away_query(match.AwayTeamId, match.Date),conn)
    drop_columns(df_previous_matches_away)
    df_previous_matches_away_agg=process_matches_average(df_previous_matches_away)

    df_previous_matches_direct_home = pd.read_sql(last_direct_home_query(match.HomeTeamId, match.AwayTeamId, match.Date) ,conn)
    drop_columns(df_previous_matches_direct_home)
    df_previous_matches_direct_agg_home=process_matches_average(df_previous_matches_direct_home)

    df_previous_matches_direct_away = pd.read_sql(last_direct_home_query(match.AwayTeamId, match.HomeTeamId, match.Date) ,conn)
    drop_columns(df_previous_matches_direct_away)
    df_previous_matches_direct_agg_away=process_matches_average(df_previous_matches_direct_away)

    print("{} {} - {} {} {} {} ".format(
        match.Date,
        match.match_result, 
        df_previous_matches_home.shape, 
        df_previous_matches_away.shape, 
        df_previous_matches_direct_home.shape, 
        df_previous_matches_direct_away.shape))

    #result.loc[0, 'team_possession'] = np.mean(match_possession_for)
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
        df_previous_matches_home_agg[0], 
        df_previous_matches_home_agg[1], 
        df_previous_matches_home_agg[2], 
        df_previous_matches_home_agg[3], 
        df_previous_matches_home_agg[4], 
        df_previous_matches_home_agg[5], 
        df_previous_matches_away_agg[0],
        df_previous_matches_away_agg[1],
        df_previous_matches_away_agg[2],
        df_previous_matches_away_agg[3],
        df_previous_matches_away_agg[4],
        df_previous_matches_away_agg[5],
        df_previous_matches_direct_agg_home[0],    
        df_previous_matches_direct_agg_home[1],
        df_previous_matches_direct_agg_home[2],
        df_previous_matches_direct_agg_away[0],    
        df_previous_matches_direct_agg_away[1],
        df_previous_matches_direct_agg_away[2],
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
        'home1_'+ df_previous_matches_home.columns,
        'home2_'+ df_previous_matches_home.columns,
        'home3_'+ df_previous_matches_home.columns,
        'home4_'+ df_previous_matches_home.columns,
        'home5_'+ df_previous_matches_home.columns,
        'home6_'+ df_previous_matches_home.columns,
        'away1_'+ df_previous_matches_away.columns,
        'away2_'+ df_previous_matches_away.columns,
        'away3_'+ df_previous_matches_away.columns,
        'away4_'+ df_previous_matches_away.columns,
        'away5_'+ df_previous_matches_away.columns,
        'away6_'+ df_previous_matches_away.columns,
        'direct1_home_'+ df_previous_matches_direct_home.columns,
        'direct3_home_'+ df_previous_matches_direct_home.columns,
        'direct6_home_'+ df_previous_matches_direct_home.columns,
        'direct1_away_'+ df_previous_matches_direct_away.columns,
        'direct3_away_'+ df_previous_matches_direct_away.columns,
        'direct6_away_'+ df_previous_matches_direct_away.columns,
    ])
    return home_away.iloc[0]

# le = preprocessing.LabelEncoder()
# def transform_columns(df_matches):
#     for column in df_matches.columns:
#         if df_matches[column].dtype != np.number:
#             print(column + ' not a number')
#             df_matches[column] = le.fit_transform(df_matches[column].astype(str)).astype('int32')
#             print(df_matches[column])
#             #print(set(s3_train_data_filtered[column]))
#         else:
#             print(column + " a number")
#             df_matches[column] = df_matches[column].astype(np.float32)

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

match_features = df_matches.apply(lambda x: get_match_features(x), axis = 1)
print(match_features.head())
match_features.to_csv('output/features3.csv',index=False)