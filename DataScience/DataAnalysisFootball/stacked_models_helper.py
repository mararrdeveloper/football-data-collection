#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from helpers import test_clfs, load_data,get_baseline, print_results, _naive_predictor

def get_predictor_of_wrong_predictions(clf):
    target_name = "wrong_pred"    
    naive_predictor, df = _naive_predictor()
    
    df['B365_pred']=naive_predictor
    df[target_name]=df['winner']!=df['B365_pred']
    df.drop(inplace=True,labels=['winner'],axis=1)
    
    target=df[target_name]
    df = pd.get_dummies(df)
    X=df.drop(target_name,axis=1)
    
    res=test_clfs(clfs=[clf],X=X,target=target,cv=10)
    
    models =[]
    for k in res.keys():
        models.append(k)
    return models[0], X, target

    
def get_predictor_on_wrong(clf):
    target_name = "winner"
    naive_predictor, df = _naive_predictor()   
    
    df['B365_pred']=naive_predictor
    df_incorrect=df[df['winner']!=df['B365_pred']]
    df_incorrect.drop(inplace=True,labels=['player0_B365','player0_Aces','player0_PS',
    'player0_EX','player0_LB','player1_B365','player1_Aces','player1_PS','player1_EX','player1_LB','B365_pred'],axis=1)
    
    target=df_incorrect[target_name]
    df_incorrect = pd.get_dummies(df_incorrect)
    X=df_incorrect.drop(labels=target_name,axis=1)
    
    res=test_clfs(clfs=[clf],X=X,target=target,cv=10)
     
    models =[]
    for k in res.keys():
        models.append(k)
    return models[0], X, target