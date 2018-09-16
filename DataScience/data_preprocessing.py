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

def get_last_matches(date, team, x = 10):
    ''' Get the last x matches of a given team. '''
    #Filter team matches from matches
    team_matches = df_matches[(df_matches['HomeTeam'] == team) | (df_matches['AwayTeam'] == team)].drop_duplicates(['ExternalId'])
    team_matches['ExternalId'] = team_matches['ExternalId'].astype(int)
    #Filter x last matches from team matches
    last_matches = team_matches[team_matches.Date < date].sort_values(by = 'Date', ascending = False).iloc[0:x,:]
    return last_matches

last_matches = get_last_matches("2018-08-27 15:00:00", "Arsenal")
last_matches.head()
def get_full_name(name):
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
    
def get_match_features(match, x = 10):
    ''' Create match specific features for a given match. '''
    
    #Define variables
    date = match.Date
    home_team = get_full_name(match.HomeTeam)
    away_team = get_full_name(match.AwayTeam)
    print(date)
    #Get last x matches of home and away team
    matches_home_team = get_last_matches(date, home_team, x = 10)
    matches_away_team = get_last_matches(date, away_team, x = 10)
    
    goals_for_home = goals_against_home = corners_for_home = corners_against_home = shotson_for_home = shotson_against_home = shotsoff_for_home = shotsoff_against_home = 0
    
    for index, row in matches_home_team.iterrows():
        isHome = True if  row['HomeTeam'] == home_team else False
        teamId = row['HomeTeamId'] if isHome else row['AwayTeamId']
        teamAgainstId = row['HomeTeamId'] if not isHome else row['AwayTeamId']

        match_goals_for = df_goals[(df_goals['TeamId'] == teamId) & (df_goals['MatchId'] == row['ExternalId'])]
        goals_for_home += match_goals_for.shape[0]
        
        match_goals_against = df_goals[(df_goals['TeamId'] == teamAgainstId) & (df_goals['MatchId'] == row['ExternalId'])]
        goals_against_home += match_goals_against.shape[0]
        
        match_corners_for = df_corners[(df_corners['TeamId'] == teamId) & (df_corners['MatchId'] == row['ExternalId'])]
        corners_for_home += match_corners_for.shape[0]
        
        match_corners_against = df_corners[(df_corners['TeamId'] == teamAgainstId) & (df_corners['MatchId'] == row['ExternalId'])]
        corners_against_home += match_corners_against.shape[0]
        
        match_shotson_for = df_shots_on[(df_shots_on['TeamId'] == teamId) & (df_shots_on['MatchId'] == row['ExternalId'])]
        shotson_for_home += match_shotson_for.shape[0]
        
        match_shotson_against = df_shots_on[(df_shots_on['TeamId'] == teamAgainstId) & (df_shots_on['MatchId'] == row['ExternalId'])]
        shotson_against_home += match_shotson_against.shape[0]
        
        match_shotsoff_for = df_shots_off[(df_shots_off['TeamId'] == teamId) & (df_shots_off['MatchId'] == row['ExternalId'])]
        shotsoff_for_home += match_shotson_for.shape[0]
        
        match_shotsoff_against = df_shots_off[(df_shots_off['TeamId'] == teamAgainstId) & (df_shots_off['MatchId'] == row['ExternalId'])]
        shotsoff_against_home += match_shotson_against.shape[0]
        
        match_possession_for = df_possessions[(df_possessions['MatchId'] == row['ExternalId'])]['HomePossession']
        match_possession_against = df_possessions[(df_possessions['MatchId'] == row['ExternalId'])]['AwayPossession']
        
    goals_for_away = goals_against_away = corners_for_away = corners_against_away = shotson_for_away = shotson_against_away = shotsoff_for_away = shotsoff_against_away = 0
    for index, row in matches_away_team.iterrows():
        isHome = True if  row['HomeTeam'] == away_team else False
        teamId = row['HomeTeamId'] if isHome else row['AwayTeamId']
        teamAgainstId = row['HomeTeamId'] if not isHome else row['AwayTeamId']

        match_goals_for = df_goals[(df_goals['TeamId'] == teamId) & (df_goals['MatchId'] == row['ExternalId'])]
        goals_for_away += match_goals_for.shape[0]
        
        match_goals_against = df_goals[(df_goals['TeamId'] == teamAgainstId) & (df_goals['MatchId'] == row['ExternalId'])]
        goals_against_away += match_goals_against.shape[0]
        
        match_corners_for = df_corners[(df_corners['TeamId'] == teamId) & (df_corners['MatchId'] == row['ExternalId'])]
        corners_for_away += match_corners_for.shape[0]
        
        match_corners_against = df_corners[(df_corners['TeamId'] == teamAgainstId) & (df_corners['MatchId'] == row['ExternalId'])]
        corners_against_away += match_corners_against.shape[0]
        
        match_shotson_for = df_shots_on[(df_shots_on['TeamId'] == teamId) & (df_shots_on['MatchId'] == row['ExternalId'])]
        shotson_for_away += match_shotson_for.shape[0]
        
        match_shotson_against = df_shots_on[(df_shots_on['TeamId'] == teamAgainstId) & (df_shots_on['MatchId'] == row['ExternalId'])]
        shotson_against_away += match_shotson_against.shape[0]
        
        match_shotsoff_for = df_shots_off[(df_shots_off['TeamId'] == teamId) & (df_shots_off['MatchId'] == row['ExternalId'])]
        shotsoff_for_away += match_shotson_for.shape[0]
        
        match_shotsoff_against = df_shots_off[(df_shots_off['TeamId'] == teamAgainstId) & (df_shots_off['MatchId'] == row['ExternalId'])]
        shotsoff_against_away += match_shotson_against.shape[0]
        
        match_possession_for = df_possessions[(df_possessions['MatchId'] == row['ExternalId'])]['HomePossession']
        match_possession_against = df_possessions[(df_possessions['MatchId'] == row['ExternalId'])]['AwayPossession']
    
        #print("for " + str(match_goals_for.shape[0]))
        #print("against " +  str(match_goals_against.shape[0]))

        #print("for " + str(np.mean(match_possession_for)))
        #print("against " +  str(np.mean(match_possession_against)))
#     #Get last x matches of both teams against each other
#     last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x = 3)
    
    result = pd.DataFrame()
    
   # result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_id'] = match.Div

#     #Create match features
    result.loc[0, 'home_team_goals_for'] = goals_for_home
    result.loc[0, 'home_team_goals_against'] = goals_against_home
    
    result.loc[0, 'home_team_corners_for'] = corners_for_home
    result.loc[0, 'home_team_corners_against'] = corners_against_home
    
    result.loc[0, 'home_team_shotson_for'] = shotson_for_home
    result.loc[0, 'home_team_shotson_against'] = shotson_against_home
    
    result.loc[0, 'home_team_shotsoff_for'] = shotsoff_for_home
    result.loc[0, 'home_team_shotsoff_against'] = shotsoff_against_home
    
    result.loc[0, 'away_team_goals_for'] = goals_for_away
    result.loc[0, 'away_team_goals_against'] = goals_against_away
    
    result.loc[0, 'away_team_corners_for'] = corners_for_away
    result.loc[0, 'away_team_corners_against'] = corners_against_away
    
    result.loc[0, 'away_team_shotson_for'] = shotson_for_away
    result.loc[0, 'away_team_shotson_against'] = shotson_against_away
    
    result.loc[0, 'away_team_shotsoff_for'] = shotsoff_for_away
    result.loc[0, 'away_team_shotsoff_against'] = shotsoff_against_away
#     result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
#     result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team) 
#     result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
#     result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
#     result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)
    return result.iloc[0]

files=[2016,2017,2018]
alldata=[]

for f in files:
    matches_with_odds = pd.read_csv(data_folder + str(f)+'.csv')
    matches_with_odds['Date']=pd.to_datetime(matches_with_odds['Date'])
    match_stats = matches_with_odds.apply(lambda x: get_match_features(x, x = 10), axis = 1)
    alldata.append(pd.concat([matches_with_odds, match_stats], axis=1))

data=pd.concat(alldata,axis=0)
data['IsTraining'] = True
data.to_csv(data_folder + '/data.csv',index=False)

matches_to_predict = pd.read_csv(data_folder + 'to_predict.csv')
matches_to_predict['Date']=pd.to_datetime(matches_to_predict['Date'])
matches_to_predict_with_stats = matches_to_predict.apply(lambda x: get_match_features(x, x = 10), axis = 1)
alldata = []
alldata.append(pd.concat([matches_to_predict, matches_to_predict_with_stats], axis=1))
data=pd.concat(alldata,axis=0)
data['IsTraining'] = False

previous_data = pd.read_csv(data_folder + '/data.csv')
data = previous_data.append(data)
data.to_csv(data_folder + '/predict_stats.csv',index=False)

