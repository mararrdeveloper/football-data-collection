import pandas as pd
import numpy as np
import helpers
from helpers import calibrate_train_clfs, load_data, get_baseline, print_results, run_clf

#X,target=load_data(columns_to_drop=['FTHG', 'FTAG',	'HTHG',	'HTAG',	'HTR'])
keep_columns =  ['IsTraining','FTR', 'Div', 'Date', 'HomeTeam', 'AwayTeam', 
    'away_away_team_corners_against', 'away_away_team_corners_for',	'away_away_team_goals_against',	'away_away_team_goals_for',	
    'away_away_team_possession',	'away_away_team_shotsoff_against',	'away_away_team_shotsoff_for',	'away_away_team_shotson_against',
    	'away_away_team_shotson_for',	'away_direct_team_corners_against',	'away_direct_team_corners_for',	'away_direct_team_goals_against',
        'away_direct_team_goals_for',	'away_direct_team_possession',	'away_direct_team_shotsoff_against',	'away_direct_team_shotsoff_for',
        'away_direct_team_shotson_against',	'away_direct_team_shotson_for',	'home_direct_team_corners_against',	'home_direct_team_corners_for',	
        'home_direct_team_goals_against',	'home_direct_team_goals_for',	'home_direct_team_possession',	'home_direct_team_shotsoff_against',
        'home_direct_team_shotsoff_for',	'home_direct_team_shotson_against',	'home_direct_team_shotson_for',	
        'home_home_team_corners_against',	'home_home_team_corners_for',	'home_home_team_goals_against',	'home_home_team_goals_for',	
        'home_home_team_possession',	'home_home_team_shotsoff_against',	'home_home_team_shotsoff_for',	'home_home_team_shotson_against', 'home_home_team_shotson_for',
        'Referee'
]

drop_columns = [
    'FTR',
    'B365H', 'B365D', 'B365A',
    'AC', 'AF', 'AR', 'AS', 'AST', 'AY', 
    'BWA', 'BWD', 'BWH', 'Bb1X2', 'BbAH', 'BbAHh', 'BbAv<2.5', 'BbAv>2.5', 'BbAvA', 'BbAvAHA', 
    'BbAvAHH', 'BbAvD', 'BbAvH', 'BbMx<2.5', 'BbMx>2.5', 'BbMxA', 'BbMxAHA', 'BbMxAHH', 'BbMxD',
    'BbMxH', 'BbOU', 'FTAG', 
    'HC', 'HF', 'HR', 'HS', 'HST', 
    'HTAG', 'HTHG', 'HTR', 'HY', 'IWA', 'IWD', 'IWH', 
    'LBA', 'LBD', 'LBH', 'PSA', 'PSCA', 'PSCD', 'PSCH', 'PSD', 'PSH', 
    'VCA', 'VCD', 'VCH', 'WHA', 'WHD', 'WHH', 'GoalFirstHalf', 'SHHG', 'SHAG', 'GoalSecondHalf','BothToScore',
]
target_name1 = "FTR"
drop_columns.remove(target_name1)

X,y = load_data(columns_to_drop=drop_columns, target_name=target_name1, is_training=True)
#X = X[X.columns.drop(list(X.filter(regex='_1_')))]

# X=X.values
# y=y.values
# y = [1 if x else 0 for x in y]
#print(X.shape)
#y = pd.DataFrame(y)
res=calibrate_train_clfs(helpers.clfs,X,y)

get_baseline()
print_results(res)

print()

clf_key = list(res.keys())[0]
clf = res[clf_key][5]
dm = res[clf_key][6]
prediction = run_clf(clf, dm, drop_columns)
print(prediction)