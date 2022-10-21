import pandas as pd
import numpy as np
from scipy import stats as st
import requests
import os
import math
import random as random
import sys


class Model:

  def __init__(self):
    self.df = pd.DataFrame
    self.predictions = []
    self.labels = []
    self.mlp_init = []
    
  def run(self, input_size, hidden_sizes, output_size):  
    #Initilize the network to have random weights between 0 and 1.

    mlp_init = []   #The network

    #Input
    hidden_nodes = []
    for i in range(input_size):
      hidden_node = []
      for i in range(hidden_sizes[0]):
        hidden_node.append(random.random())
      hidden_nodes.append(hidden_node)

    mlp_init.append(hidden_nodes)

    #Init number of weights between each hidden layer
    for sizes in range(len(hidden_sizes)-1):
      hidden_nodes = []
      for i in range(hidden_sizes[1+sizes]):
        hidden_node = []
        for i in range(hidden_sizes[0+sizes]):
          hidden_node.append(random.random())
        hidden_nodes.append(hidden_node)

      mlp_init.append(hidden_nodes)

    #output layer
    output_nodes = []
    for i in range(hidden_sizes[-1]):
      output_node = []
      for i in range(output_size):
        output_node.append(random.random())
      output_nodes.append(output_node)
    
    
    mlp_init.append(output_nodes)

    self.mlp_init = mlp_init

  def forwardProp(self,input):
    values = [[]]
    values[0] = input
    #loops through each layer
    for i in range(len(self.mlp_init)):
      layerValues = []
      #loops through each node
      for j in range(len(self.mlp_init[i])):
        value = float()
        #loops through each weight
        for k in range(len(self.mlp_init[i][j])):
          value = value + (float(values[i][k]) * float(self.mlp_init[i][j][k]))
        value = 1/(1 + math.e**(-value))
        layerValues.append(value)
      values.append(layerValues)

    print("Values: ", values)




modeling = Model()
modeling.run(2, [2,3,2], 2)
input = [2,4]
modeling.forwardProp(input)
