import pandas as pd
import numpy as np
import helpers
from helpers import test_clfs, load_data, get_baseline, print_results, _naive_predictor, _data_load_helper
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

target_name = "FTR"

naive_predictor, df = _naive_predictor()

df['B365_pred']=naive_predictor
df_incorrect=df[df['FTR']!=df['B365_pred']]
#df[target_name]=df['FTR']!=df['B365_pred']
df_incorrect=df_incorrect[['B365H', 'B365D','B365A','home_team_goals_for', 'home_team_goals_against', 
    'home_team_corners_for', 'home_team_corners_against', 'home_team_shotson_for', 'home_team_shotson_against', 
    'home_team_shotsoff_for', 'home_team_shotsoff_against', 'away_team_goals_for', 'away_team_goals_against', 
    'away_team_corners_for', 'away_team_corners_against', 'away_team_shotson_for', 'away_team_shotson_against', 
    'away_team_shotsoff_for', 'away_team_shotsoff_against','FTR']]
df_incorrect.to_csv('data/temp_data.csv')
target=df_incorrect[target_name]

df_incorrect.drop(inplace=True,labels=['FTR'],axis=1)
#df_incorrect = pd.get_dummies(df_incorrect)
X = df_incorrect

print(len(df_incorrect.columns))


#X=df_incorrect.drop(target_name,axis=1)

res=test_clfs(clfs=helpers.clfs,X=X,target=target,cv=10)


print("frequency of wrong predictions is: "+str((df_incorrect.shape[0])/df.shape[0]))

print_results(res)
