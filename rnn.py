#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:46:49 2017

@author: eduardopoleo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Use Scaling by normalization Min
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)#=> [[1], [2], [3]]

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) #=> price at [day1, day2...day60]
    y_train.append(training_set_scaled[i, 0]) #=> price at day60

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# This is to add the extra dimention for extra indicators, which in this case
# is 1 (google stock historical prices)
# keras expect this shape [ [[1...60], [60...99]], [[aaa....hhh], [hhh...zzz]] ]
# where each middle array contains the different sequential set of data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# number of units
# return_sequences (cuz we'll be retuning values)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# How many neurons are going to be ignore with each iteration. E.g 20%
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

# Compiling the RNN
# mean squared error since it's not a classification problem
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Save weights
regressor.save_weights('weights.h5')

# Load weights
regressor.load_weights('weights.h5')

# Get test dataset
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values



# Getting predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# make all values to be understood as separete rows.
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0]) #=> price at [day1, day2...day60]

X_test = np.array(X_test)
# 3d structure
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.xlabel('Stock Price')
plt.legend()
plt.show()


# How to improve this model.

#Getting more training data: we trained our model on the past 5 years of the 
#Google Stock Price but it would be even better to train it on the past 10 years.

#Increasing the number of timesteps: the model remembered the stock prices from 
#the 60 previous financial days to predict the stock price of the next day. 
#That’s because we chose a number of 60 timesteps (3 months). You could try to 
#increase the number of timesteps, by choosing for example 120 timesteps (6 months).

#Adding some other indicators: if you have the financial instinct that the 
#stock price of some other companies might be correlated to the one of Google, 
#you could add this other stock price as a new indicator in the training data.

#Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
#Adding more neurones in the LSTM layers: we highlighted the fact that we needed 
#a high number of neurones in the LSTM layers to respond better to the complexity 
#of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. 
#You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.
