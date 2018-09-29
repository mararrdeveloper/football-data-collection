#DB Connection 
import pymssql
import pandas as pd
import pyodbc
import numpy as np
from random import random

data_folder = 'DataAnalysisFootball/data/'

conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                     "Server=Martin-PC\SQLEXPRESS;"
                     "Database=FootballData;"
                     "Trusted_Connection=yes;")
#conn = pymssql.connect(server="localhost", user="",password="", port=63642)

stmt = """SELECT  Teams.FullName as HomeTeam, Teams2.FullName as AwayTeam, 
        Teams.ExternalId as HomeTeamId, Teams2.ExternalId as AwayTeamId
	  ,[Matches].ExternalId as ExternalId
      ,[Date]
      ,[Country]
      ,[League]
      ,[Season]
      ,[Stage]
      ,[AwayTeam_Id]
      ,[HomeTeam_Id]
FROM  [FootballData].[dbo].[Matches]
LEFT JOIN [Teams] ON Matches.HomeTeam_Id = Teams.Id
LEFT JOIN [Teams] as Teams2 ON Matches.AwayTeam_Id = Teams2.Id"""

# Excute Query here
df_matches = pd.read_sql(stmt,conn)
df_matches.drop_duplicates(['ExternalId'], inplace=True)
df_matches['Date']=pd.to_datetime(df_matches['Date'])
print(df_matches.shape)

stmt = "SELECT * FROM Goals"
df_goals = pd.read_sql(stmt,conn)
df_goals.drop('Id', inplace=True, axis=1)
df_goals.drop_duplicates(inplace=True)
print(df_goals.shape)

stmt = "SELECT * FROM Corners"
df_corners = pd.read_sql(stmt,conn)
df_corners.drop_duplicates(['ExternalId'], inplace=True)
df_corners.head(2)
print(df_corners.shape)

stmt = "SELECT * FROM ShotOns"
df_shots_on = pd.read_sql(stmt,conn)
df_shots_on.drop_duplicates(['ExternalId'], inplace=True)
print(df_shots_on.shape)

#Shots off select
stmt = "SELECT * FROM ShotOffs"
df_shots_off = pd.read_sql(stmt,conn)
df_shots_off.drop_duplicates(['ExternalId'], inplace=True)
print(df_shots_off.shape)

#Possessions select
stmt = "SELECT * FROM Possessions"
df_possessions = pd.read_sql(stmt,conn)

df_possessions.drop_duplicates(['ExternalId'], inplace=True)
df_possessions.replace('', np.nan, inplace=True)
df_possessions.dropna(axis=0, how='any', inplace=True)
print(df_possessions.shape)

df_possessions['HomePossession'] = df_possessions['HomePossession'].astype(int)
df_possessions['AwayPossession'] = df_possessions['AwayPossession'].astype(int)
df_possessions['Minute'] = df_possessions['Minute'].astype(int)

stmt = "SELECT * FROM Teams"
df_teams = pd.read_sql(stmt,conn)
df_teams.drop_duplicates(['ExternalId'], inplace=True)
print(df_teams.shape)

def get_last_n_matches(team_matches, date, last_n):
    last_matches = team_matches[team_matches.Date <= date].sort_values(by = 'Date', ascending = False).iloc[0:last_n,:] 
    return last_matches

def get_matches_team(date, team, last_n, is_home):
    ''' Get the last x matches of a given team. '''
    #Filter team matches from matches
    if is_home:
        team_matches = df_matches[(df_matches['HomeTeam'] == team)].drop_duplicates(['ExternalId'])
    else:
        team_matches = df_matches[(df_matches['AwayTeam'] == team)].drop_duplicates(['ExternalId'])
        
    team_matches['ExternalId'] = team_matches['ExternalId'].astype(int)
    last_matches = get_last_n_matches(team_matches, date, last_n)

    if len(last_matches) == 0:
        print("No matches for " + team)
    return last_matches

def get_last_direct_matches(date, team_home, team_away, last_n):
    direct_matches = df_matches[((df_matches['HomeTeam'] == team_home) & (df_matches['AwayTeam'] == team_away)) |
        (df_matches['HomeTeam'] == team_away) & (df_matches['AwayTeam'] == team_home)]

    direct_matches['ExternalId'] = direct_matches['ExternalId'].astype(int)
    last_matches = get_last_n_matches(direct_matches, date, last_n)
    
    if len(last_matches) == 0:
        print("No direct matches for " + team_home + " " +team_away)
    return last_matches
#last_matches = get_last_matches("2018-08-27 15:00:00", "Arsenal", 15)
#last_matches.head()


def get_full_name(name):
    #missmapics of the 2 databases
    name_dict = {
        "Man United" : "Manchester United",
        "Man City" : "Manchester City",
        "Wolves" : "Wolverhampton Wanderers",
        "Newcastle" : "Newcastle United",
        "West Brom" : "West Bromwich Albion",
        "QPR" : "Queens Park Rangers",
    }
    if name in name_dict:
        return name_dict[name]
    return name

def get_match_features(match):
    ''' Create match specific features for a given match. '''
    all_data = pd.DataFrame()
    #Get last x matches of home and away team
    home_team_name = get_full_name(match.HomeTeam)
    away_team_name = get_full_name(match.AwayTeam)

    matches_home_team_home = get_matches_team(match.Date, home_team_name, 15, True) 
    home_team_home = process_matches_average(matches_home_team_home, home_team_name)

    #matches_home_team_away = get_matches_team(match.Date, home_team_name, 15, False) 
    #home_team_away = process_matches_average(matches_home_team_away, home_team_name)
   
    #matches_away_team_home = get_matches_team(match.Date, away_team_name, 15, True)
    #away_team_home = process_matches_average(matches_away_team_home, away_team_name)

    matches_away_team_away = get_matches_team(match.Date, away_team_name, 15, False)
    away_team_away = process_matches_average(matches_away_team_away, away_team_name)

    direct_matches = get_last_direct_matches(match.Date, home_team_name, away_team_name, 15)
    
    home_team_direct = process_matches_average(direct_matches, home_team_name)
    away_team_direct = process_matches_average(direct_matches, away_team_name)

    data=[home_team_home.loc['average'], home_team_direct.loc['average'], away_team_away.loc['average'], away_team_direct.loc['average']]
    
    data=np.hstack(data)
    home_away=pd.DataFrame(data)
    home_away=home_away.transpose()
    home_away.columns=np.hstack([
        'home_home_'+ home_team_home.columns,
        'home_away_'+ home_team_direct.columns,
        'away_away_'+ away_team_away.columns,
        'away_home_'+ away_team_direct.columns
        ])
    
    print(str(match.Date) + " " + home_team_name + " " + away_team_name)
    return home_away.iloc[0]

def process_matches_average(matches, team_name):
    home_team_data = pd.DataFrame(index=['average'])
    for index, row in matches.iterrows():
        match_id = row['ExternalId']
        is_home = True if row['HomeTeam'] == team_name else False

        home_team_id = row['HomeTeamId']
        away_team_id = row['AwayTeamId']
        #print(home_team_id + " " + away_team_id)

        home_team_match_data = get_team_features(match_id, home_team_id, is_home)
        home_team_data = home_team_data.append(home_team_match_data)
    
    if len(home_team_data) > 0:
        home_team_data.loc['average'] = home_team_data.sum().div(len(matches))

    return home_team_data

def get_team_features(match_id, team_id, is_home):
    result = pd.DataFrame()
    #Create match features)
    match_goals_for = df_goals[(df_goals['TeamId'] == team_id) & (df_goals['MatchId'] == match_id)]
    result.loc[0, 'team_goals_for'] = match_goals_for.shape[0]
    match_goals_against = df_goals[(df_goals['TeamId'] != team_id) & (df_goals['MatchId'] == match_id)]
    result.loc[0, 'team_goals_against'] = match_goals_against.shape[0]
    
    match_corners_for = df_corners[(df_corners['TeamId'] == team_id) & (df_corners['MatchId'] == match_id)]
    result.loc[0, 'team_corners_for'] = match_corners_for.shape[0]
    match_corners_against = df_corners[(df_corners['TeamId'] != team_id) & (df_corners['MatchId'] == match_id)]
    result.loc[0, 'team_corners_against'] = match_corners_against.shape[0]

    match_shotson_for = df_shots_on[(df_shots_on['TeamId'] == team_id) & (df_shots_on['MatchId'] == match_id)]
    result.loc[0, 'team_shotson_for'] = match_shotson_for.shape[0]
    match_shotson_against = df_shots_on[(df_shots_on['TeamId'] != team_id) & (df_shots_on['MatchId'] == match_id)]
    result.loc[0, 'team_shotson_against'] = match_shotson_against.shape[0]  

    match_shotsoff_for = df_shots_off[(df_shots_off['TeamId'] == team_id) & (df_shots_off['MatchId'] == match_id)]
    result.loc[0, 'team_shotsoff_for'] = match_shotsoff_for.shape[0]
    match_shotsoff_against = df_shots_off[(df_shots_off['TeamId'] != team_id) & (df_shots_off['MatchId'] == match_id)]
    result.loc[0, 'team_shotsoff_against'] = match_shotsoff_against.shape[0]

    if is_home:
        match_possession_for = df_possessions[(df_possessions['MatchId'] == match_id)]['HomePossession']
        result.loc[0, 'team_possession'] = np.mean(match_possession_for)
    else:
        match_possession_for = df_possessions[(df_possessions['MatchId'] == match_id)]['AwayPossession']
        result.loc[0, 'team_possession'] = np.mean(match_possession_for)
    
    return result.iloc[0]

files=[2016, 2017, 2018]
alldata=[]

for f in files:
    matches_with_odds = pd.read_csv(data_folder + str(f)+'.csv')
    matches_with_odds['Date']=pd.to_datetime(matches_with_odds['Date'])
    match_stats = matches_with_odds.apply(lambda x: get_match_features(x), axis = 1)
    alldata.append(pd.concat([matches_with_odds, match_stats], axis=1))

data=pd.concat(alldata,axis=0)
data['IsTraining'] = True
data.to_csv(data_folder + '/data.csv',index=False)

matches_to_predict = pd.read_csv(data_folder + 'to_predict.csv')
matches_to_predict['Date']=pd.to_datetime(matches_to_predict['Date'])
matches_to_predict_with_stats = matches_to_predict.apply(lambda x: get_match_features(x), axis = 1)
alldata = []
alldata.append(pd.concat([matches_to_predict, matches_to_predict_with_stats], axis=1)[matches_to_predict.columns.tolist() + matches_to_predict_with_stats.columns.tolist()])
data=pd.concat(alldata,axis=0)
data['IsTraining'] = False

previous_data = pd.read_csv(data_folder + '/data.csv')
data = previous_data.append(data)
data.to_csv(data_folder + '/predict_stats_odds_1.csv',index=False)

