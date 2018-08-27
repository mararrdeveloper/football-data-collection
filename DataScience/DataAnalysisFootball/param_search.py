#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from random import uniform

import pandas as pd
import numpy as np

from helpers import test_clfs, load_data,get_baseline, print_results, _naive_predictor
  

X, target = load_data()

#Classifier definition    

clf1 = RandomForestClassifier()
clf2 = LogisticRegression()
clf3 = KNeighborsClassifier()
clf4 = ExtraTreesClassifier()
clf5 = GradientBoostingClassifier()

clfs=[clf1, clf2, clf3, clf4, clf5]
names=["Random Forest", "Logistic Regression",'kNN','ExtraTrees', "GradientBoost"]

#number of iterations per classifier
iters=[30,10,10,30,30]


#Classifier parameters
param_trees = {"n_estimators": sp_randint(20,1000),
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(2, 11),
              "criterion": ["gini", "entropy"]}

param_LR = {"C":sp_randint(1,20)}

param_knn = {'n_neighbors':sp_randint(2,15)}

param_boost = {"n_estimators":sp_randint(50,500),'max_depth':sp_randint(1,4),'subsample':[0.8,0.9,1.0]}

params_list = [param_trees, param_LR, param_knn, param_trees, param_boost]


#the code for the actual search

#res is going to be the dictionary where we store results
res={}
for clf,name,n_iter,params in zip(clfs,names,iters,params_list):
    print('\n\n\ntesting classifier : ' + name +"\n")
    if n_iter>0:                   
        random_search = RandomizedSearchCV(clf, param_distributions=params,
                                       n_iter=n_iter,verbose=0)
        
        random_search.fit(X,target)        
        #store all the results  in a dictionary
        res[random_search] = cross_val_score(random_search,X,target,cv=10)
        
    
print_results(res)

