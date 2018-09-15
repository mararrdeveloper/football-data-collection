import pandas as pd
import numpy as np
import helpers

from helpers import test_clfs, load_data, print_results, get_baseline

X,target=load_data(columns_to_keep=['league_id', 'home_team_goals_for', 'home_team_goals_against', 
    'home_team_corners_for', 'home_team_corners_against', 'home_team_shotson_for', 'home_team_shotson_against', 
    'home_team_shotsoff_for', 'home_team_shotsoff_against', 'away_team_goals_for', 'away_team_goals_against', 
    'away_team_corners_for', 'away_team_corners_against', 'away_team_shotson_for', 'away_team_shotson_against', 
    'away_team_shotsoff_for', 'away_team_shotsoff_against','FTR'], target_name='FTR')

res=test_clfs(clfs=helpers.clfs,X=X,target=target,cv=10)

get_baseline()

print_results(res)
