from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

clfs = [#LogisticRegression(),  
        # RandomForestClassifier(n_estimators=150),
         ExtraTreesClassifier(n_estimators=50),
        # KNeighborsClassifier(5),
        # SVC(kernel="linear", C=0.025,degree=2),
        # SVC(gamma=2, C=1, probability=True),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
        # MLPClassifier(hidden_layer_sizes=(20, ),solver='lbfgs')
        ]

def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (accuracy_score(y, yPred), 
            precision_score(y, yPred, pos_label=3, average='macro'), 
            recall_score(y, yPred, pos_label=3, average='macro'))

def test_clfs(clfs,X,target,cv=10,scoring="accuracy"):
    res = defaultdict(list)
    for clf in clfs:
     
        print("testing : "+str(clf))
        scores = cross_val_score(clf, X, target, cv=cv,scoring=scoring)
        print(scores)
        # PCA
        # X = StandardScaler().fit_transform(X)
        # pca = PCA(n_components=12)
        # X = pca.fit_transform(X, y=target)
        X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.20)

        clf_fit = clf.fit(X_train, target_train)
        
        #from sklearn import preprocessing
        #lb = preprocessing.LabelBinarizer()
        #target = lb.fit(target)

        prediction = clf.predict(X_test)
        #print(prediction)
        accuracy = accuracy_score(target_test, prediction)
        precision = precision_score(target_test, prediction, average=None)
        recall = recall_score(target_test, prediction, average=None)
      
        confusion = confusion_matrix(target_test, prediction)  # 
        #plt.imshow(confusion, cmap='binary', interpolation='None')
        #plt.show()

        res[clf_fit].append(scores)
        res[clf_fit].append(accuracy)
        res[clf_fit].append(precision)
        res[clf_fit].append(recall)
        res[clf_fit].append(confusion)
        res[clf_fit].append(clf)
        #print(accuracy)
        #print(precision)
        #print(recall)
        #print(confusion)
        #y_predict_test = clf.predict(X_test)
        #F1_score
        #score_test = metrics.f1_score(y_test, y_predict_test, pos_label=list(set(y_test)), average = None)
    return res


def load_data(columns_to_keep=None,columns_to_drop=None,scaler=StandardScaler(),target_name="FTR", is_training=True):
    df =_data_load_helper()

    return _preprocess_data(df,columns_to_keep,columns_to_drop,scaler,target_name, is_training)
    
def _data_load_helper(data_path = "data/predict_stats_odds_1.csv"):
    df = pd.read_csv(data_path)

    df = df.dropna(axis=0, how='any')
    df = df.reset_index()
    return df
    
def _preprocess_data(df,columns_to_keep=None,columns_to_drop=None,scaler=StandardScaler(),target_name="FTR", is_training=True):
    #turn a month into a string in order to dummify it
    #df.month=str(df.month)
    #df['HomeTeam'] = pd.Categorical(df.HomeTeam)
    #df['AwayTeam'] = pd.Categorical(df.AwayTeam)
    print("KEEEP")
    #print(columns_to_keep)
    df = df.dropna(subset=['B365H','B365A','B365D'])
    if columns_to_keep!=None:
        df=df[columns_to_keep]
        print(df.shape)
    elif columns_to_drop!=None:
        
        df=df.drop(labels=columns_to_drop,axis=1)
    
    
    if is_training:
        targets = df[df['IsTraining'] == True]
    else:
        targets = df[df['IsTraining'] == False]

    target=targets[target_name]
    length = len(target)

    df=df.drop(target_name,axis=1)
    df=df.drop("IsTraining",axis=1)
    #print(df.columns)        
    df = pd.get_dummies(df)
    
    X=df
    #choices for scalers are StandardScaler and MinMaxScaler
 
    #print(X.head())
    X=scaler.fit_transform(X)
    
    print("LENGTH" + str(length))
    if is_training:
        X = X[:length]
    else:
        X = X[-length:]

    return X, target
    

def _naive_predictor():
    df=_data_load_helper()
    naive_predictor=[]
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
    for key,item in res.items():  
        cross_val_results = item[0]
        print("{0}\n mean: {1}, std: {2}".format(str(key)[:22],str(np.mean(cross_val_results)),str(np.std(cross_val_results))))
        print("accuracy: {0},\nprecission: {1},\n recall:   {2}, \n confusion:".format(str(item[1]), str(item[2]), str(item[3])))
        print(item[4])

def run_clf(clf, columns):
    X, target = load_data(is_training=False, columns_to_keep=columns)
    #print(X)
    print(X.shape)
    #print(X)
    prediction = clf.predict(X)
    return prediction