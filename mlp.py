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
    self.output = 0
    
  def run(self, input_size, hidden_sizes, output_size,flag):  
    #Initilize the network to have random weights between 0 and 1.

    mlp_init = []   #The network

    if flag == 0:

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

    else:

      #Input
      hidden_nodes = []
      for i in range(input_size):
        hidden_node = []
        for i in range(output_size):
          hidden_node.append(random.random())
        hidden_nodes.append(hidden_node)

      mlp_init.append(hidden_nodes)
      

      #output layer
      #output_nodes = []
      #for i in range(input_size):
      #  output_node = []
      #  for i in range(output_size):
      #    output_node.append(random.random())
      #  output_nodes.append(output_node)
      
      
      #mlp_init.append(output_nodes)


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
        
        #print(len(values[i]))

        #loops through each node according to the next layer length (ex. 0,1,2,3)  
        for j in range(len(self.mlp_init[i+1])): 
          l = []
          
          #print('j :',j)

          #This will grab everything in values starting at values[0]
          for k in range(len(values[i])): 
            #print('Current posistion in forwards prop',i,k,j)
            #print(float(values[i][k]))
            #print(float(self.mlp_init[i][k][j]))
            l.append(float(values[i][k])*float(self.mlp_init[i][k][j]))  #do xiwi
          summation = sum(l) #Sum of all xiwis
          
          #print('Sum xi*wi : ',summation)
          

          sigmoid = 1/(1+(math.e**(-summation)))    #sigmoid function

          #print('After sigmoidal activation : ',sigmoid)
          layer_outputs.append(sigmoid) #append for each input

        #print('Entire Layer output : ',layer_outputs)

        values.append(layer_outputs) #append all the outputs. (this will be what is "inside" of each node)

      

      #output layer
      elif classNumber == 1:

        #Linear = No activation
        l = []
        for k in range(len(values[-1])):  
          l.append(float(values[-1][k])*float(self.mlp_init[-1][k][0]))  #do xiwi
        summation = sum(l)
        output = summation

        layer_outputs = []
        layer_outputs.append(output)
        values.append(layer_outputs)

        self.output = output

        #print('Before linear activation : ', values[-2])
       
        #print('Linear Output : ', self.output)


        self.values = values

        #print('Output Values for layers 3->1 : ',self.values)
        #print('Output at layer 0 : ',self.output)


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

        #print('weights', self.mlp_init[-1])
        #print("Before Softamx : " ,values[-1])

        values[-1] = output_values

        #print('After Softmax : ',values[-1])

      
    self.values = values
    

  def Back_Prop(self,eta,classNumber,actual,output_size):  
    
    #Preapre Deltas for delta rule
    deltas=[]  
    for x in range(len(self.values)):
      deltas.append([])

    counter = 0
    #go through every layer backwards (ex. 3,2,1,0)
    for i in reversed(range(len(self.values))):   #we need #of deltas = # of layers so len delta = len values.... Used to be len deltas = len self.mlp_intit

      #farthest_layer_right = self.mlp_init[0]

      if i == len(self.values)- 1:    #output layer
        if classNumber == 1:
          diff = actual - self.output    #(t - a)    t = actual a = guess
          deltas[counter].append(diff)   
          

        else:
          actualClass = actual[1]
          actualOneHot = actual[0]

          for j in range(len(self.values[i])):
            diff = actualOneHot.get(actualClass)[j] - self.values[i][j]
            deltas[0].append(diff)

      else:
        #print(counter)

        #Go through every node in the current Layer and assign each one a delta
        for j in range(len(self.values[i])): 

          xi = self.values[i][j]
          sumwih_deltai = 0

          

          #This grabs how many nodes are in the layer ahead. That is how many components will be in each delta calc. (ex. 0,1)
          for l in range(len(deltas[counter-1])):

            deltai = deltas[counter-1][l]
            weight_s = self.mlp_init[i][j][l]  

            #get sum(wiDi)
            sumwih_deltai += weight_s*deltai
          
          #Get delta for the given node.
          delcalc = sumwih_deltai*(xi)*(1-xi) 
          deltas[counter].append(delcalc)
 




    

      counter = counter + 1  

    #print('Before weights update : ',self.mlp_init)

    deltas.reverse()
    #print(deltas)

    #Now we need to update the weights
    #go through every layer
    for i in range(len(self.mlp_init)):
      layer = self.mlp_init[i]
      #go through every node in every layer
      for j in range(len(layer)):
        neuron = layer[j]
        #go through every weight
        for k in range(len(neuron)):
          #print(self.mlp_init[i][j][k])
          #print(self.values[i][j])

          #should be every weight  + eta*delta in from of the weight*xi that caused the weight
          self.mlp_init[i][j][k] = self.mlp_init[i][j][k] + eta*deltas[i+1][k]*self.values[i][j]    #delta needs to be +1 so we do not pull from the input layer

    #print('After weights update : ',self.mlp_init)
    
  def change_mlp_init(self, change):
    self.mlp_init = change
    
  def algGA(self):
    pass

  def algGE(self):
    pass

  def algPSO(self, particles):
    #----Hyper-Parameters----
    omega = 0.1
    c_1 = 1.496
    c_2 = 1.496

    #----Variables----
    parts = particles
    r_1 = 0
    r_2 = 0
    numPart = 8
    loops = 10
    pb = []

    for i in range(loops):
      for j in range(numPart):
        if i == 0 and j == 0:
          for k in range(numPart):
            pass
            #parts[k].forwardProp(#FINISH THIS)
