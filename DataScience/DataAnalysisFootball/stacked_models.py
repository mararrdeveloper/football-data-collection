#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from helpers import test_clfs, load_data, get_baseline, print_results, _naive_predictor, _data_load_helper
from stacked_models_helper import get_predictor_of_wrong_predictions, get_predictor_on_wrong
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

clf1 = KNeighborsClassifier(5)
clf2 = LogisticRegression()

model1, X1, target1 = get_predictor_of_wrong_predictions(clf2)
preds1 = cross_val_predict(model1, X1, target1, cv=10)


final_pred=[]
model2, X2, target2 = get_predictor_on_wrong(clf2)
preds2 = cross_val_predict(model2, X2, target2, cv=10)

iter_values = X1.drop(labels=['player0_B365','player0_Aces','player0_PS',
    'player0_EX','player0_LB','player1_B365','player1_Aces','player1_PS','player1_EX',
    'player1_LB','B365_pred'],axis=1).values
             
#Go through each row and if classifier1 predicts B365 is going to get it wrong
#then use the second model to make a prediction (1 or 0). Otherwise append -2.
for pred,datapoint in zip(preds1,iter_values):
    if pred==True:
        final_pred.append(int(model2.predict(datapoint.reshape(1, -1))))
    else:
        final_pred.append(-2)

final_pred=np.array(final_pred)

print(len(final_pred[final_pred>0]))
print(len(final_pred))


#simulation
df = load_data()
total=0
bet=1

for i in range(0,len(target1)):
    if final_pred[i]>=0:
        if final_pred[i]==target1.iloc[i]:
            if target1.iloc[i]==1:
                #we are subtracting the bet, in order to calculate the pure profit
                payout=X1.iloc[i]['player1_B365']*bet - bet
            elif target1.iloc[i]==0:
                payout=X1.iloc[i]['player0_B365']*bet - bet
        else:
                payout= -1.0*bet
        total+=payout

print(total)