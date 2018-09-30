from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.pipeline import Pipeline

from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time


clfs = [#LogisticRegression(),  
        # RandomForestClassifier(n_estimators=200),
        # ExtraTreesClassifier(n_estimators=50),
        # KNeighborsClassifier(5),
        # SVC(kernel="linear", C=0.025,degree=2),
        # SVC(gamma=2, C=1, probability=True),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1),
        # AdaBoostClassifier(),
        GaussianNB(),
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
    feature_len = X.shape[1]
    parameters_GNB = {'dm_reduce__n_components': np.arange(5, feature_len, int(np.around(feature_len/5)))}

    for clf in clfs:
        
        #Initializing dimensionality reductions
        pca = PCA()
        dm_reductions = [pca]  

        print("testing : "+str(clf))
        scores = cross_val_score(clf, X, target, cv=cv,scoring=scoring)
        print(scores)
        # PCA
        # X = StandardScaler().fit_transform(X)
        # pca = PCA(n_components=12)
        # X = pca.fit_transform(X, y=target)

        #X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.20)
        #clf_fit = clf.fit(X_train, target_train)
        scorer = make_scorer(accuracy_score)

        X_train_calibrate, X_test, y_train_calibrate, y_test = train_test_split(X, target, test_size=0.20)
        X_train, X_calibrate, y_train, y_calibrate = train_test_split(X_train_calibrate, y_train_calibrate, test_size = 0.3, random_state = 42)

        #Creating cross validation data splits
        cv_sets = model_selection.StratifiedShuffleSplit(n_splits = 5, test_size = 0.20, random_state = 5)
        cv_sets.get_n_splits(X_train, y_train)

        clf, dm_reduce, train_score, test_score = train_calibrate_predict(clf = clf, dm_reduction = pca, X_train = X_train, y_train = y_train,
                    X_calibrate = X_calibrate, y_calibrate = y_calibrate,
                    X_test = X_test, y_test = y_test, cv_sets = cv_sets,
                    params = parameters_GNB, scorer = scorer, jobs = 1, use_grid_search = True)

        
        
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()

          # PCA
        #X_test = StandardScaler().fit_transform(X_test)
        #pca = PCA(n_components = 5)
        #X_test = pca.fit_transform(X_test, y=y_test)
        #target = lb.fit(y_test)
        clf_fit = clf.fit(dm_reduce.transform(X_train), y_train)
        #y_test = y_test.apply(pd.to_numeric, errors='coerce')
        prediction = clf.predict(dm_reduce.transform(X_test))
        #print(prediction)
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average=None)
        recall = recall_score(y_test, prediction, average=None)
      
        confusion = confusion_matrix(y_test, prediction)  # 
        #plt.imshow(confusion, cmap='binary', interpolation='None')
        #plt.show()

        res[clf_fit].append(scores)
        res[clf_fit].append(accuracy)
        res[clf_fit].append(precision)
        res[clf_fit].append(recall)
        res[clf_fit].append(confusion)
        res[clf_fit].append(clf)
        res[clf_fit].append(dm_reduce)
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

    df = df.fillna(0)
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

def run_clf(clf, dm, drop_columns):
    X, target = load_data(is_training=False, columns_to_drop=drop_columns)
    #print(X)
    print(X.shape)
    #print(X)
    prediction = clf.predict(dm.transform(X))
    return prediction

def train_calibrate_predict(clf, dm_reduction, X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, cv_sets, params, scorer, jobs, 
                            use_grid_search = True, **kwargs):
    ''' Train and predict using a classifer based on scorer. '''
    
    #Indicate the classifier and the training set size
    print("Training a {} with {}...".format(clf.__class__.__name__, dm_reduction.__class__.__name__))
    
    #Train the classifier
    best_pipe = train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs)
    
    #Calibrate classifier
    print("Calibrating probabilities of classifier...")
    start = time()    
    clf = CalibratedClassifierCV(best_pipe.named_steps['clf'], cv= 'prefit', method='isotonic')
    clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
    end = time()
    print("Calibrated {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start)/60))
    
    # Print the results of prediction for both training and testing
    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__, predict_labels(clf, best_pipe, X_train, y_train)))
    print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__, predict_labels(clf, best_pipe, X_test, y_test)))
    
    #Return classifier, dm reduction, and label predictions for train and test set
    return clf, best_pipe.named_steps['dm_reduce'], predict_labels(clf, best_pipe, X_train, y_train), predict_labels(clf, best_pipe, X_test, y_test)

def predict_labels(clf, best_pipe, features, target):
    ''' Makes predictions using a fit classifier based on scorer. '''
    
    #Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(best_pipe.named_steps['dm_reduce'].transform(features))
    print(target.values)
    print("actual")
    print(y_pred)
    end = time()
    
    #Print and return results
    print("Made predictions in {:.4f} seconds".format(end - start))
    return accuracy_score(target.values, y_pred)

def train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs, use_grid_search = True, 
                     best_components = None, best_params = None):
    ''' Fits a classifier to the training data. '''
    
    #Start the clock, train the classifier, then stop the clock
    start = time()
    
    #Check if grid search should be applied
    if use_grid_search == True: 
        
        #Define pipeline of dm reduction and classifier
        estimators = [('dm_reduce', dm_reduction), ('clf', clf)]
        pipeline = Pipeline(estimators)
        
        #Grid search over pipeline and return best classifier
        grid_obj = model_selection.GridSearchCV(pipeline, param_grid = params, scoring = scorer, cv = cv_sets, n_jobs = jobs)
        grid_obj.fit(X_train, y_train)
        best_pipe = grid_obj.best_estimator_
    else:
        
        #Use best components that are known without grid search        
        estimators = [('dm_reduce', dm_reduction(n_components = best_components)), ('clf', clf(best_params))]
        pipeline = Pipeline(estimators)        
        best_pipe = pipeline.fit(X_train, y_train)
        
    end = time()
    
    #Print the results
    print("Trained {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start)/60))
    
    #Return best pipe
    return best_pipe