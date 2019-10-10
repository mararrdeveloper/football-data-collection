
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
from helpers import calibrate_train_clfs, load_data, get_baseline, print_results, run_clf


# In[36]:


keep_columns =  ['IsTraining','FTR', 'Div', 'Date', 'HomeTeam', 'AwayTeam', 
    'away_away_team_corners_against', 'away_away_team_corners_for',	'away_away_team_goals_against',	'away_away_team_goals_for',	
    'away_away_team_possession',	'away_away_team_shotsoff_against',	'away_away_team_shotsoff_for',	'away_away_team_shotson_against',
    	'away_away_team_shotson_for',	'away_direct_team_corners_against',	'away_direct_team_corners_for',	'away_direct_team_goals_against',
        'away_direct_team_goals_for',	'away_direct_team_possession',	'away_direct_team_shotsoff_against',	'away_direct_team_shotsoff_for',
        'away_direct_team_shotson_against',	'away_direct_team_shotson_for',	'home_direct_team_corners_against',	'home_direct_team_corners_for',	
        'home_direct_team_goals_against',	'home_direct_team_goals_for',	'home_direct_team_possession',	'home_direct_team_shotsoff_against',
        'home_direct_team_shotsoff_for',	'home_direct_team_shotson_against',	'home_direct_team_shotson_for',	
        'home_home_team_corners_against',	'home_home_team_corners_for',	'home_home_team_goals_against',	'home_home_team_goals_for',	
        'home_home_team_possession',	'home_home_team_shotsoff_against',	'home_home_team_shotsoff_for',	'home_home_team_shotson_against',
                 'home_home_team_shotson_for','Referee' 
]

drop_columns = ['B365H', 'B365D', 'B365A',
    'AC', 'AF', 'AR', 'AS', 'AST', 'AY', 'BWA', 'BWD', 'BWH', 'Bb1X2', 'BbAH', 'BbAHh', 'BbAv<2.5', 'BbAv>2.5', 'BbAvA', 'BbAvAHA', 
    'BbAvAHH', 'BbAvD', 'BbAvH', 'BbMx<2.5', 'BbMx>2.5', 'BbMxA', 'BbMxAHA', 'BbMxAHH', 'BbMxD', 'BbMxH', 'BbOU', 'FTAG', 'FTHG', 'HC', 'HF', 'HR', 'HS', 'HST', 
    'HTAG', 'HTHG', 'HTR', 'HY', 'IWA', 'IWD', 'IWH', 'LBA', 'LBD', 'LBH', 'PSA', 'PSCA', 'PSCD', 'PSCH', 'PSD', 'PSH', 
    'VCA', 'VCD', 'VCH', 'WHA', 'WHD', 'WHH'
]

X,y = load_data(columns_to_drop=drop_columns, is_training=True)


# In[37]:


X.shape


# In[38]:


y.shape


# In[39]:


import numpy
import pandas
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Bidirectional, LSTM, TimeDistributed
from keras.initializers import glorot_normal

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# In[40]:


encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[58]:
# =============================================================================
# 
# # create a sequence classification instance
# def get_sequence(n_timesteps):
# 	# create a sequence of random numbers in [0,1]
# 	X = array([random() for _ in range(n_timesteps)])
# 	# calculate cut-off value to change class values
# 	limit = n_timesteps/4.0
# 	# determine the class outcome for each item in cumulative sequence
# 	y = array([0 if x < limit else 1 for x in cumsum(X)])
# 	# reshape input and output data to be suitable for LSTMs
# 	X = X.reshape(1, n_timesteps, 1)
# 	y = y.reshape(1, n_timesteps, 1)
# 	return X, y
# 
# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(LSTM(20, input_shape=(5, 1), return_sequences=True))
#     #model.add(Lambda(lambda x: tf.expand_dims(model.output, axis=-1)))
# #    model.add(Dense(20,  activation='relu')) # input_dim=314,
#     #model.add(Bidirectional(LSTM(256,  return_sequences=True, dropout=0.1, recurrent_dropout=0.1,kernel_initializer=glorot_normal(seed=None)),name = 'BDLSTM1')) #Best = 300,0.25,0.25
#     #model.add(TimeDistributed(Dense(1, activation='sigmoid')))
#     model.add(TimeDistributed(Dense(1, activation='softmax')))
#     #model.add(model.add(TimeDistributed(Dense(1, activation='softmax'))))
#     # Compile model
#     
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# =============================================================================
from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

# create a sequence classification instance
def get_sequence(n_timesteps, X, y):
	# create a sequence of random numbers in [0,1]
	#X = array([random() for _ in range(n_timesteps)])
    
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	#y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y

# define problem properties
# n_timesteps = 10
# # define LSTM
# model = Sequential()
# model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True))
# model.add(TimeDistributed(Dense(1, activation='sigmoid')))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# # train LSTM
# for epoch in range(1000):
# 	# generate new random sequence
# 	X,y = get_sequence(n_timesteps, X, y)
# 	# fit model for one epoch on this sequence
# 	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# # evaluate LSTM
# X,y = get_sequence(n_timesteps, X, y)
# yhat = model.predict_classes(X, verbose=1)
# for i in range(n_timesteps):
# 	print('Expected:', y[0, i], 'Predicted', yhat[0, i])

# # In[59]:


#seed = 7
# #numpy.random.seed(seed)
# from sklearn.model_selection import TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=2)# max_train_size=100
# for train_index, test_index in tscv.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#kfold = KFold(n_splits=10)
# shuffle=True, random_state=seed
results = cross_val_score(estimator, X, y, cv=tscv)


# In[54]:


results


# In[55]:


import statistics 
x = statistics.mean(results)
x


# In[45]:


X_train_calibrate, X_test, y_train_calibrate, y_test = train_test_split(X, y, test_size=0.20)
X_train, X_calibrate, y_train, y_calibrate = train_test_split(X_train_calibrate, y_train_calibrate, test_size = 0.3, random_state = 42)

