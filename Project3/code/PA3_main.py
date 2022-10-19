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
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/loss.py?raw=true', 'loss.py']]


for i in range(len(classInputArray)):
  with open(classInputArray[i][1], 'w') as f:
    r = requests.get(classInputArray[i][0])
    f.write(r.text)

import preprocessing as pp
import loss as lss

#------------------------Main--------------------------

def main():

#-----Hyper_parameters-------#

  k_nn = 7
  k_clusters = 30
  slct = 4
  sigma = 40
  epsilon = 0.1

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

    #print(preProcess.folds())

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
