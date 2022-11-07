#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: Neural Net
##Completed: 10-9-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import numpy as np
from scipy import stats as st
import requests
import os
import math
import random as random
import sys

#----------------------classes-------------------------

class Model:

  def __init__(self):
    self.df = pd.DataFrame
    self.predictions = []
    self.labels = []
    self.mlp_init = []
    self.values = []
    
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
      for i in range(hidden_sizes[0+sizes]):
        hidden_node = []
        for j in range(hidden_sizes[1+sizes]):
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

    #print(mlp_init)

    

    self.mlp_init = mlp_init

    #print(self.mlp_init)

    '''

    mlp_init is set up in the following fashion.

    input layer ->    input_nodes[weight_going_out,weight_going_out,weight_going_out ... weight_going_out]
    hiden layer 1 ->  hidden_nodes[weight_going_out,weight_going_out,weight_going_out ... weight_going_out]
    ...
    hideen layer x -> hidden_nodes[weight_going_out,weight_going_out,weight_going_out ... weight_going_out]
    output layer ->   ouput_nodes[weight_going_out,weight_going_out,weight_going_out ... weight_going_out]

    '''
    
  def forwardProp(self,input, classNumber):
    values = [[]]
    values[0] = input


    #loops through each layer.  (ex. 0,1,2,3)
    for i in range(len(self.mlp_init)):
      layer_outputs = []
 
      if i != len(self.mlp_init)-1: #As long as we are not in the last layer

        #loops through each node according to the next layer length (ex. 0,1,2,3)  
        for j in range(len(self.mlp_init[i+1])): 
          l = []

          #This will grab everything in values starting at values[0]
          for k in range(len(values[i])): 
            l.append(float(values[i][k])*float(self.mlp_init[i][k][j]))  #do xiwi
          summation = sum(l) #Sum of all xiwis

          sigmoid = 1/(1+(math.e**(-summation)))    #sigmoid function
          layer_outputs.append(sigmoid) #append for each input

        values.append(layer_outputs) #append all the outputs. (this will be what is "inside" of each node)

      #output layer
      elif classNumber == 1:

        #Linear = No activation
        l = []
        for k in range(len(values[-1])):  
          l.append(float(values[-1][k])*float(self.mlp_init[-1][k][0]))  #do xiwi
        summation = sum(l)
        output = summation

        self.output = output


      else:
        layer_outputs = []
        for i in range(len(self.mlp_init[-1][0])):
 
          l = []
          for k in range(len(values[-1])):   #for every xi
            l.append(float(values[-1][k])*float(self.mlp_init[-1][k][i]))  #do xiwi
          summation = sum(l) #Sum of all xiwis


          sigmoid = 1/(1+math.e**(-summation))    #sigmoid function
          layer_outputs.append(sigmoid) 

        values.append(layer_outputs)
      
      if classNumber == 0:

        sum_of_soft = []
        for i in values[-1]:
          softmax1 = (math.e**i)
          sum_of_soft.append(softmax1)
          
        the_sum_of_soft = sum(sum_of_soft)

        output_values = []

        for i in values[-1]:
          softmax2 = (math.e**i)/the_sum_of_soft
          output_values.append(softmax2)

        values[-1] = output_values

      
    self.values = values

  def Back_Prop(self,eta,classNumber,actual,output_size):  

    deltas=[]  
    for x in range(len(self.mlp_init)):
      deltas.append([])

 

    counter = 0
    for i in reversed(range(len(self.mlp_init))):  
      farthest_layer_right = self.mlp_init[i]    

      if i == len(self.mlp_init)- 1:    
 
        if classNumber == 1:
          print('hello')
          diff =  self.output -actual            

          deltas[counter].append(diff)
          

        else:
          actualClass = actual[1]
          actualOneHot = actual[0]

          for j in range(len(self.values[i])):
            diff = actualOneHot.get(actualClass)[j] - self.values[i][j]
            deltas[0].append(diff)

      else:

  
        for j in range(len(farthest_layer_right)): 

          print('right layer',len(farthest_layer_right))




          xi = self.values[i][j] 

          sumwih_deltai = 0




          for l in range(len(deltas[counter])):


            weight_s = self.mlp_init[i][j][l]


            print(len(self.mlp_init[i][j]))
            print(deltas)
            print(counter)

            deltai = deltas[counter][l]




            a_sum = weight_s*deltai

            sumwih_deltai = sumwih_deltai + a_sum


          delcalc = sumwih_deltai*(xi)*(1-xi) 


          deltas[counter+1].append(delcalc)
 


        

        counter = counter + 1  

    deltas.reverse()


    for i in range(len(self.mlp_init)):
      layer = self.mlp_init[i]
      for j in range(len(layer)):
        neuron = layer[j]
        for k in range(len(neuron)):

          if j < len(self.values[i]):

            self.mlp_init[i][j][k] = self.mlp_init[i][j][k] - eta*deltas[i][j]*self.values[i][j]
