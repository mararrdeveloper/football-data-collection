import pandas as pd
import numpy as np
import helpers
from helpers import test_clfs, load_data, get_baseline, print_results, run_clf

#X,target=load_data(columns_to_drop=['FTHG', 'FTAG',	'HTHG',	'HTAG',	'HTR'])
columns =  ['IsTraining','FTR', 'Div', 'Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'home_team_goals_for', 'home_team_goals_against', 'home_team_corners_for', 
    'home_team_corners_against', 'home_team_shotson_for', 'home_team_shotson_against', 'home_team_shotsoff_for', 'home_team_shotsoff_against', 'away_team_goals_for', 
    'away_team_goals_against', 'away_team_corners_for', 'away_team_corners_against', 'away_team_shotson_for', 'away_team_shotson_against', 'away_team_shotsoff_for', 'away_team_shotsoff_against']
X,target = load_data(columns_to_keep=columns, is_training=True)
print(X.shape)
res=test_clfs(clfs=helpers.clfs,X=X,target=target,cv=10)

get_baseline()
print_results(res)

print()

clf_key = list(res.keys())[0]
clf = res[clf_key][5]
prediction = run_clf(clf, columns)
print(prediction)