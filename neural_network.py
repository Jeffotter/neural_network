# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:00:10 2020

@author: Charles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

#this is where i bring in my data to use for the neural network. 
#So far I cannot get it to predict arsenic concentrations from field parameters accurately
df = pd.read_csv(r'C:\Users\Charles\Documents\MF_analytical_data_AND_field_Params.csv')
As = df['Arsenic, Dissolved']
del df['Location'], df['CollectDate'], df['Unnamed: 0']
#df = df[['ORP','pH','Cond','Turb','DO']]

###############################################################################

def sigmoid(x):
    return expit(x)

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],self.input.shape[0]) #equal to [columns of X, rows of X]
        self.weights2   = np.random.rand(self.input.shape[0]) #equal to [rows of X]
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    #define your X and Y here (neural network will use )
    X = np.array(df)
    y = np.array(As)
    
    nn = NeuralNetwork(X,y)

    for i in range(1000):
        nn.feedforward()
        nn.backprop()

###############################################################################

    #the better the neural network is at predicting y the closer to a 1:1 ratio this plot should have
    print(nn.output)
    plt.scatter(nn.output,y)