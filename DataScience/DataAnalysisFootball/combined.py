import pandas as pd
import numpy as np
import helpers
from helpers import test_clfs, load_data, get_baseline, print_results, run_clf

#X,target=load_data(columns_to_drop=['FTHG', 'FTAG',	'HTHG',	'HTAG',	'HTR'])
keep_columns =  ['IsTraining','FTR', 'Div', 'Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A',
    'away_away_team_corners_against', 'away_away_team_corners_for',	'away_away_team_goals_against',	'away_away_team_goals_for',	
    'away_away_team_possession',	'away_away_team_shotsoff_against',	'away_away_team_shotsoff_for',	'away_away_team_shotson_against',
    	'away_away_team_shotson_for',	'away_direct_team_corners_against',	'away_direct_team_corners_for',	'away_direct_team_goals_against',
        'away_direct_team_goals_for',	'away_direct_team_possession',	'away_direct_team_shotsoff_against',	'away_direct_team_shotsoff_for',
        'away_direct_team_shotson_against',	'away_direct_team_shotson_for',	'home_direct_team_corners_against',	'home_direct_team_corners_for',	
        'home_direct_team_goals_against',	'home_direct_team_goals_for',	'home_direct_team_possession',	'home_direct_team_shotsoff_against',
        'home_direct_team_shotsoff_for',	'home_direct_team_shotson_against',	'home_direct_team_shotson_for',	
        'home_home_team_corners_against',	'home_home_team_corners_for',	'home_home_team_goals_against',	'home_home_team_goals_for',	
        'home_home_team_possession',	'home_home_team_shotsoff_against',	'home_home_team_shotsoff_for',	'home_home_team_shotson_against', 'home_home_team_shotson_for'
]

drop_columns = [
    'AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'BWA', 'BWD', 'BWH', 'Bb1X2', 'BbAH', 'BbAHh', 'BbAv<2.5', 'BbAv>2.5', 'BbAvA', 'BbAvAHA', 
    'BbAvAHH', 'BbAvD', 'BbAvH', 'BbMx<2.5', 'BbMx>2.5', 'BbMxA', 'BbMxAHA', 'BbMxAHH', 'BbMxD', 'BbMxH', 'BbOU', 'FTAG', 'FTHG', 'HC', 'HF', 'HR', 'HS', 'HST', 
    'HTAG', 'HTHG', 'HTR', 'HY', 'IWA', 'IWD', 'IWH', 'LBA', 'LBD', 'LBH', 'PSA', 'PSCA', 'PSCD', 'PSCH', 'PSD', 'PSH', 'Referee', 
    'VCA', 'VCD', 'VCH', 'WHA', 'WHD', 'WHH'
]

X,target = load_data(columns_to_drop=drop_columns, is_training=True)
print(X.shape)
res=test_clfs(clfs=helpers.clfs,X=X,target=target,cv=10)

get_baseline()
print_results(res)

print()

clf_key = list(res.keys())[0]
clf = res[clf_key][5]
dm = res[clf_key][6]
prediction = run_clf(clf, dm, drop_columns)
print(prediction)