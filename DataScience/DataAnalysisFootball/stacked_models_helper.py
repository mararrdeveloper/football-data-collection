#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from helpers import test_clfs, load_data,get_baseline, print_results, _naive_predictor

def get_predictor_of_wrong_predictions(clf):
    target_name = "wrong_pred"    
    naive_predictor, df = _naive_predictor()
    
    df['B365_pred']=naive_predictor
    df[target_name]=df['FTR']!=df['B365_pred']
    df.drop(inplace=True,labels=['FTR'],axis=1)
    
    target=df[target_name]
    df = pd.get_dummies(df)
    X=df.drop(target_name,axis=1)
    df.head()
    res=test_clfs(clfs=[clf],X=X,target=target,cv=10)
    
    models =[]
    for k in res.keys():
        models.append(k)
    return models[0], X, target

    
def get_predictor_on_wrong(clf):
    target_name = "FTR"
    naive_predictor, df = _naive_predictor()   
    
    df['B365_pred']=naive_predictor
    df_incorrect=df[df['FTR']!=df['B365_pred']]
    # df_incorrect.drop(inplace=True,labels=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 
    # 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 
    # 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 
    # 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 
    # 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'PSCH', 'PSCD', 'PSCA',
    # 'home_team_goals_for', 'home_team_goals_against', 
    # 'home_team_corners_for', 'home_team_corners_against', 'home_team_shotson_for', 'home_team_shotson_against', 
    # 'home_team_shotsoff_for', 'home_team_shotsoff_against', 'away_team_goals_for', 'away_team_goals_against', 
    # 'away_team_corners_for', 'away_team_corners_against', 'away_team_shotson_for', 'away_team_shotson_against', 
    # 'away_team_shotsoff_for', 'away_team_shotsoff_against'],axis=1)

    target=df_incorrect[target_name]
    
    df_incorrect=df_incorrect.drop(labels=target_name,axis=1)
    X = pd.get_dummies(df_incorrect)

    res=test_clfs(clfs=[clf],X=X,target=target,cv=10)
     
    models =[]
    for k in res.keys():
        models.append(k)
    return models[0], X, target