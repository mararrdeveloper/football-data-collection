from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

clfs = [LogisticRegression(),  
        RandomForestClassifier(n_estimators=150),
        ExtraTreesClassifier(n_estimators=50),
        KNeighborsClassifier(5),
        #SVC(kernel="linear", C=0.025,degree=2),
        SVC(gamma=2, C=1, probability=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(hidden_layer_sizes=(20, ),solver='lbfgs')]

def test_clfs(clfs,X,target,cv=10,scoring="accuracy"):
    res={}
    
    for clf in clfs:
        print("testing : "+str(clf))
        scores = cross_val_score(clf, X, target, cv=cv,scoring=scoring)
        res[clf.fit(X, target)]=scores
    return res

def _data_load_helper():
    df = pd.read_csv("data/data.csv")
    print(df.columns)
    df = df.dropna(axis=0, how='all')
    df = df.reset_index()
    #print(len(df))
    return df
    
def _preprocess_data(df,columns_to_keep=None,columns_to_drop=None,scaler=StandardScaler(),target_name="FTR"):
    #turn a month into a string in order to dummify it
    #df.month=str(df.month)
    #df['HomeTeam'] = pd.Categorical(df.HomeTeam)
    #df['AwayTeam'] = pd.Categorical(df.AwayTeam)
    df = df.dropna(subset=['B365H','B365A','B365D'])
    if columns_to_keep!=None:
        df=df[columns_to_keep]
    elif columns_to_drop!=None:
        df=df.drop(labels=columns_to_drop,axis=1)
 
    target=df[target_name]
    df=df.drop(target_name,axis=1)
    print(df.columns)        
    df = pd.get_dummies(df)
    
    X=df
    #choices for scalers are StandardScaler and MinMaxScaler
    X=scaler.fit_transform(X)
    
    return X, target
    
def load_data(columns_to_keep=None,columns_to_drop=None,scaler=StandardScaler(),target_name="FTR"):
    df=_data_load_helper()
    return _preprocess_data(df,columns_to_keep,columns_to_drop,scaler,target_name)
    
def _naive_predictor():
    df=_data_load_helper()
    naive_predictor=[]
    print(len(df))
    for i in range(0, len(df)):
        row=df.iloc[i]
        if int(row['B365H'])<= int(row['B365A']):
            naive_predictor.append('H')
        else:
            naive_predictor.append('A')
            
    return naive_predictor,df
    
def get_baseline():
    naive_predictor,df = _naive_predictor() 
    print(len(naive_predictor==df['FTR'])) 
    print(len(df))
    print('Bet 365 predictor: ' + str(sum(naive_predictor==df['FTR'])*1.0/df.shape[0]))
    print('Majority predictor: '+str(sum(df['FTR']=='H')*1.0/df.shape[0]))
    
def print_results(res):
    #display the mean and the standard deviation of the folds from each run
    for i,item in enumerate(res.items()):
        print("classifier {0} mean: {1}, std: {2}".format(str(i),str(np.mean(item[1])),str(np.std(item[1]))))