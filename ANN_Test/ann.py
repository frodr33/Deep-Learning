# Artificial Neural Network

# Installing Theano Open Source numerical computation library, based on numpi
# syntax. Runs on both CPU and also GPU (which is a processor for grpahics purposes)
# GPU can do more floating point calculations and parallel computations
# GPU much better of Neural Networks
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# --upgrade --no-deps git+git://github.com/Theano/Theano.git


# Installing Tensorflow CPU/GPU devloped by Google team, used for research
# and development purposes. 
# pip install tensorflow

# Installing Keras. Warps Tensorflow and Theano so we dont have to build neural
# networks from scratch
# pip install --upgrade keras

#####################################################################
######################Part 1 - Data Preprocessing####################
#####################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # Independent Variable, col 3..12, only include
                                 # variables that have impact on outcome
y = dataset.iloc[:, 13].values # Dependent Variable
 
'''
Have all the independent variables but do not have the weights on them!
This is the job of the artifical neural network with back propagation
'''

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # Encoding Country, fit_transform converts strings to numbers
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # Encoding Gender
onehotencoder = OneHotEncoder(categorical_features = [1]) # Dummy Variable
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # remove first of 3 dummy variables to avoid "trap" ???

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#####################################################################
#################Part 2 - Now let's make the ANN!####################
#####################################################################

# keras a neural network library
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # initialize Neural Network
from keras.layers import Dense # Build layers of ANN

# Initialising the ANN, is Sequence of Layers
classifier = Sequential() 

'''
Steps in building ANN
    1). Randomly initalize weights to small numbers, done by Dense
    2). Put all Independent variables into input layer
    3). Neuron applies activation function, most common is rectifier and sigmoid
    4). Compare predicted result to actual 
    5). Error is backpropagated, and then weights updated 
    6). Repeat
'''

# Adding the input layer and the first hidden layer
'''
units -> number of nodes in hidden layer, 
    tip: avg of # of nodes in input and output
    When output layer has binary number, only has one node
kernel_initializer (uniform function) -> initalizes weights 
    according to uniform distribution
activation -> 'relu' = rectifier function
input_dim -> number of inputs, a.k.a independent variables


'''
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# Using sigmoid function allows for calculting probabilties
# If you have output with more than 2 categories, must input 3 and use
# activation of softmax which is just sigmoid applied to systems of more
# than 3 categories
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
'''
optimizer -> alogirthm you want to update weights,
     'adam' = Stochastic Gradient Descent
loss -> loss function within adam alogrithm, must be optimized
      to find optimal weights (Sum of squared difference between pred. and act.)
      'binary_crossentropy' == Logarithmic Loss Function for binary outcomes
      'categorical_crossentropy' == Log loss function for more than 2 outcomes
metrics -> use 'Accuracy' to improve ANN's performance (need to put it in list)
'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
'''
batch_size -> Update weights after every batch of size 10
epochs -> How many loops are done over all the data?
'''
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#####################################################################
########Part 3 - Making predictions and evaluating the model#########
#####################################################################
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # make it into a boolean array

# Making the Confusion Matrix, allows you see how many were right
# and how many were wrong
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




# Essential Library
#   -Keras
#   -Tensorflow --> Very complicated ... which is why keras wraps theano and tensorflow
#   -Numpy
#   -Sklearn

# Recurrrent Neural Networks
#   - Used for Time Series Analysis
#   - ANN is for Regression and Classification
#   - CNN used for computer vision






















