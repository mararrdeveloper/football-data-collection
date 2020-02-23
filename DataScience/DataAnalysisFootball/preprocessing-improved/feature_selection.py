import pymssql
import pandas as pd
import pyodbc
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from random import random
from sql.sql_statements import select_matches_with_targets, match_aggregated_stats
#DB Connection 
conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                     "Server=Martin-PC\SQLEXPRESS;"
                     "Database=FootballData;"
                     "Trusted_Connection=yes;")

# Excute Query here
df_matches = pd.read_sql(select_matches_with_targets,conn)
df_matches.drop_duplicates(['MatchId'], inplace=True)
df_matches['Date']=pd.to_datetime(df_matches['Date'])
print(df_matches.shape)

custom_lables = ['H', 'A', 'D']
print(custom_lables)

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

    df_previous_matches_home = pd.read_sql(
        match_aggregated_stats.format(
            "6", 
            "HomeTeamId =" + str(match.HomeTeamId), 
            "", 
            match.Date)
        ,conn)
    df_previous_matches_home.drop(['MatchId','HomeTeamFullName','AwayTeamFullName','HomeTeamId','AwayTeamId','match_result','date'],axis =1,inplace=True)
    print(df_previous_matches_home.shape)

    df_previous_matches_home_sum=process_matches_average(df_previous_matches_home)
    df_previous_matches_away = pd.read_sql(
        match_aggregated_stats.format(
            "6",
            "AwayTeamId = " +  str(match.AwayTeamId), 
            "", 
            match.Date)
         ,conn)
    print(df_previous_matches_away.shape)

    df_previous_matches_away.drop(['MatchId','HomeTeamFullName','AwayTeamFullName','HomeTeamId','AwayTeamId','match_result','date'],axis =1,inplace=True)
    df_previous_matches_away_sum=process_matches_average(df_previous_matches_away)

    df_previous_matches_direct = pd.read_sql(match_aggregated_stats.format(
            "6",
            "",
            "(HomeTeamId = " + str(match.HomeTeamId) + "AND AwayTeamId = " + str(match.AwayTeamId) + ")",
            #" OR (HomeTeamId = " + str(match.AwayTeamId) + "AND AwayTeamId = " + str(match.HomeTeamId) + ")", 
            match.Date)
        ,conn)

    df_previous_matches_direct.drop(['MatchId','HomeTeamFullName','AwayTeamFullName','HomeTeamId','AwayTeamId','match_result','date'],axis =1,inplace=True)
    df_previous_matches_direct_sum=process_matches_average(df_previous_matches_direct)
    print(df_previous_matches_direct.shape)

    data=[
        match.match_result,
        match.HomeTeamFullName,
        match.AwayTeamFullname,
        df_previous_matches_home_sum, 
        df_previous_matches_away_sum,
        df_previous_matches_direct_sum,
        # home_team_home_6.loc['average'],
        # home_team_away_1.loc['average'], 
        # home_team_away_3.loc['average'],
        # home_team_away_6.loc['average'],
        # home_team_direct.loc['average'],
        # away_team_away_1.loc['average'],
        # away_team_away_3.loc['average'],
        # away_team_away_6.loc['average'],
        # away_team_home_1.loc['average'],
        # away_team_home_3.loc['average'],
        # away_team_home_6.loc['average'],
        # away_team_direct.loc['average']
        ]
    
    data=np.hstack(data)
    home_away=pd.DataFrame(data)
    home_away=home_away.transpose()
    home_away.columns=np.hstack([
        'FTR',
        'HomeTeam',
        'AwayTeam',
        'home_'+ df_previous_matches_home.columns,
        'away_'+ df_previous_matches_away.columns,
        'direct_'+ df_previous_matches_direct.columns,
    #     'home_away_1_'+ home_team_away_1.columns,
    #     'home_away_3_'+ home_team_away_3.columns,
    #     'home_away_6_'+ home_team_away_6.columns,
    #     'home_direct_'+ home_team_direct.columns,
    ])
    print(home_away.columns)
    
    #concat = pd.concat([matches_to_predict, matches_to_predict_with_stats], axis=1)[matches_to_predict.columns.tolist() + matches_to_predict_with_stats.columns.tolist()]
    #alldata.append(concat)
    #data=pd.concat(alldata,axis=0)

    
    #print(df_previous_matches.columns)
    #sum = df_previous_matches.sum()

    return home_away.iloc[0]

def process_matches_average(matches):
  
    #matches["home_team_possession_avg"] = matches["home_team_possession_avg"].mean()
    return matches.mean(axis = 0, skipna = True)

   

#matches_with_odds['Date']=pd.to_datetime(matches_with_odds['Date'])
match_features = df_matches.apply(lambda x: get_match_features(x), axis = 1)
#transform_columns(match_features)
print(match_features.head())
match_features.to_csv('output/features.csv',index=False)
# preprocess()

# def preprocess():
#     alldata=[]

#         alldata.append(pd.concat([matches_with_odds, match_stats], axis=1))

#     data=pd.concat(alldata,axis=0)
#     data['BothToScore'] 
#     data['GoalFirstHalf'] 
#     data['SHHG']
#     data['SHAG'] 
#     data['GoalSecondHalf']
    
#     data.to_csv(data_folder + '/data.csv',index=False)

#     matches_to_predict = pd.read_csv(data_folder + 'to_predict.csv')
#     matches_to_predict['Date']=pd.to_datetime(matches_to_predict['Date'])
    

#     matches_to_predict_with_stats = matches_to_predict.apply(lambda x: get_match_features(x), axis = 1)
#     alldata = []
#     alldata.append(pd.concat([matches_to_predict, matches_to_predict_with_stats], axis=1)[matches_to_predict.columns.tolist() + matches_to_predict_with_stats.columns.tolist()])
#     data=pd.concat(alldata,axis=0)
#     data['IsTraining'] = False

#     previous_data = pd.read_csv(data_folder + '/data.csv')
#     data = previous_data.append(data)
#     data.to_csv(data_folder + 'processed/features_0.csv',index=False)



# def get_team_features(match_id, team_id, is_home):
#     result = pd.DataFrame()
#     #Create match features)
#     match_goals_for = df_goals[(df_goals['TeamId'] == team_id) & (df_goals['MatchId'] == match_id)]
#     result.loc[0, 'team_goals_for'] = match_goals_for.shape[0]
#     match_goals_against = df_goals[(df_goals['TeamId'] != team_id) & (df_goals['MatchId'] == match_id)]
#     result.loc[0, 'team_goals_against'] = match_goals_against.shape[0]
    
#     match_corners_for = df_corners[(df_corners['TeamId'] == team_id) & (df_corners['MatchId'] == match_id)]
#     result.loc[0, 'team_corners_for'] = match_corners_for.shape[0]
#     match_corners_against = df_corners[(df_corners['TeamId'] != team_id) & (df_corners['MatchId'] == match_id)]
#     result.loc[0, 'team_corners_against'] = match_corners_against.shape[0]

#     # corners = df_corners[(df_corners['MatchId'] == match_id)]
#     # match_first_corner = corners['Minute'].min()
#     # result.loc[0, 'match_first_corner'] = match_first_corner
#     # print(corners.shape)
#     # print(match_first_corner)
#     # print()


#     match_shotson_for = df_shots_on[(df_shots_on['TeamId'] == team_id) & (df_shots_on['MatchId'] == match_id)]
#     result.loc[0, 'team_shotson_for'] = match_shotson_for.shape[0]
#     match_shotson_against = df_shots_on[(df_shots_on['TeamId'] != team_id) & (df_shots_on['MatchId'] == match_id)]
#     result.loc[0, 'team_shotson_against'] = match_shotson_against.shape[0]  

#     match_shotsoff_for = df_shots_off[(df_shots_off['TeamId'] == team_id) & (df_shots_off['MatchId'] == match_id)]
#     result.loc[0, 'team_shotsoff_for'] = match_shotsoff_for.shape[0]
#     match_shotsoff_against = df_shots_off[(df_shots_off['TeamId'] != team_id) & (df_shots_off['MatchId'] == match_id)]
#     result.loc[0, 'team_shotsoff_against'] = match_shotsoff_against.shape[0]

#     if is_home:
#         match_possession_for = df_possessions[(df_possessions['MatchId'] == match_id)]['HomePossession']
#         result.loc[0, 'team_possession'] = np.mean(match_possession_for)
#     else:
#         match_possession_for = df_possessions[(df_possessions['MatchId'] == match_id)]['AwayPossession']
#         result.loc[0, 'team_possession'] = np.mean(match_possession_for)
    
#     return result.iloc[0]