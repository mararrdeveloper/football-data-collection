import pandas as pd
import numpy as np
import helpers
from helpers import test_clfs, load_data, get_baseline, print_results, _naive_predictor, _data_load_helper
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

target_name = "wrong_pred"

naive_predictor, df = _naive_predictor()

df['B365_pred']=naive_predictor
df[target_name]=df['FTR']!=df['B365_pred']
df.drop(inplace=True,labels=['FTR'],axis=1)
print(len(df.columns))
target=df[target_name]
#df = pd.get_dummies(df)
X=df.drop(target_name,axis=1)

res=test_clfs(clfs=helpers.clfs,X=X,target=target,cv=10,scoring="f1")


print("frequency of wrong predictions is: "+str(1.0*sum(df['wrong_pred'])/df.shape[0]))

print_results(res)
