# -*- coding: utf-8 -*-
"""main

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m75UwxwUx82FvybX7kodza5jIjjv8PA4
"""

#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: K-nearest neighbor
##Completed: 10-9-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import requests
import os
import random

#----Python Classes import----

classInputArray = [['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/preprocessing.py?raw=true','preprocessing.py'], 
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/model.py?raw=true','model.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/model_regression.py?raw=true','model_regression.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/loss.py?raw=true', 'loss.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/k_means_classify.py?raw=true', 'k_means_classify.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/k_means_regress.py?raw=true', 'k_means_regress.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/enn_classify.py?raw=true', 'ennc.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/enn_regress.py?raw=true', 'ennr.py']]


for i in range(len(classInputArray)):
  with open(classInputArray[i][1], 'w') as f:
    r = requests.get(classInputArray[i][0])
    f.write(r.text)

import preprocessing as pp
import model as classifier
import model_regression as regressor
import loss as lss
import k_means_classify as kmeans
import k_means_regress as kmeansr
import ennc as enncc
import ennr as ennrr

#------------------------Main--------------------------

def main():

#-----Hyper_parameters-------#

  k_nn = 3
  k_clusters = 30
  slct = 5
  sigma = 10
  epsilon = 0.0001

#----------------------------#

  #runs preProcessing
  preProcess = pp.Preprocessing()
  preProcess.process()
  preProcess.fold()
  data = [preProcess.tuning, preProcess.folds]

  #runs the classification model and the loss function
  if slct == 0:
    #model 
    modeling = classifier.Model()
    modeling.run(data,k_nn)

    #loss 
    classesU = preProcess.df['Class'].unique()
    lossValues = lss.Loss()
    lossValues.calculate(classesU, modeling.predictions, modeling.labels)

    print("\n\n\n\nF1 Score: ", lossValues.F1)

  #runs the regression model and loss function
  if slct == 1:
    #model
    modeling = regressor.Model()
    modeling.run(data,k_nn)
    
    #loss
    lossValues = lss.Loss()
    lossValues.calculateReg(modeling.predictions, modeling.labels)

    print("\n\n\n\nmse Score: ", lossValues.mse, lossValues.mae)

  #runs the k-means regression model and loss function
  if slct == 2:
    #model
    modeling = kmeansr.Model()
    modeling.run(data,k_nn,k_clusters,sigma)
    
    #loss
    lossValues = lss.Loss()
    lossValues.calculateReg(modeling.predictions, modeling.labels)

    print("\n\n\n\nmse Score: ", lossValues.mse, lossValues.mae)

  #runs the k-means classification model and loss function
  if slct == 3:
    #model
    modeling = kmeans.Model()
    modeling.run(data,k_nn,k_clusters,preProcess.value)
    
    #loss
    classesU = preProcess.df['Class'].unique()
    lossValues = lss.Loss()
    lossValues.calculate(classesU, modeling.predictions, modeling.labels)

    print("\n\n\n\nF1 Score: ", lossValues.F1, "\n\nPrecision", lossValues.prec)

  #runs the categorical edited nearest neighbor model
  if slct == 4:
    #model
    modeling = enncc.Model()
    modeling.run(data,k_nn,k_clusters,preProcess.value)
    
    #loss
    classesU = preProcess.df['Class'].unique()
    lossValues = lss.Loss()
    lossValues.calculate(classesU, modeling.predictions, modeling.labels)

    print("\n\n\n\nF1 Score: ", lossValues.F1, "\n\nPrecision", lossValues.prec)

  #runs the regression edited nearest neighbor model
  if slct == 5:   
    #model
    modeling = ennrr.Model()
    modeling.run(data, k_nn, sigma, epsilon)
    
    #loss
    lossValues = lss.Loss()
    lossValues.calculateReg(modeling.predictions, modeling.labels)

    print("\n\n\n\nmse Score: ", lossValues.mse, lossValues.mae)

#calls main
main()

#makes sure that the python classes are taken out after execution
for i in range(len(classInputArray)):
  fileName = classInputArray[i][1]
  os.remove(fileName)

