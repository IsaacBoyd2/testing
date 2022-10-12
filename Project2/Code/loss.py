#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: K-nearest neighbor
##Completed: 9-29-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import math
import random
import numpy as np

#------------------------Class--------------------------
class Loss:

  #initialization
  def __init__(self):
    self.prec = []
    self.rec = []
    self.error = []
    self.F1 = int()
    self.mse = int()
    self.mae = int()
  
  def calculate(self, classes, pred, facts):


    #classes = preProcess.df['Class'].unique()


    confusionMat = np.zeros([len(classes),len(classes)])

    #populates the confusion matrix that will be used for precision, recall, and F1 calculations.
    for i in range(len(facts)):
      for j in range(len(facts[i])):
        indHorz = 0
        indVert = 0
        #checks for position in the confusion matrix
        for k in range(len(classes)):
          if classes[k] == pred[i][j]:
            indVert = k
          if classes[k] == facts[i][j]:
            indHorz = k
        #adds an occurance in the confusion matrix
        confusionMat[indHorz, indVert] += 1

    #calculates precision and recall
    for i in range(len(confusionMat)):
      truePos = 0
      falsePos = 0
      falseNeg = 0
      for j in range(len(confusionMat[i])):
        if i == j:
          truePos = confusionMat[i][j]
        else:
          falsePos = falsePos + confusionMat[i][j]
          falseNeg = falseNeg + confusionMat[j][i]

      if truePos + falsePos != 0:   
        precision = truePos/(truePos+falsePos)
        self.prec.append(precision)
        
      else:
        self.prec.append(0)

      if truePos + falseNeg != 0: 
        recall = truePos/(truePos+falseNeg)
        self.rec.append(recall)
        
      else:
        self.prec.append(0)

      
      

    #gets the average precision and recall, then calculates the F1 score
    avgPrec = 0
    avgRec = 0
    for i in range(len(self.prec)):
      avgPrec = avgPrec + self.prec[i]
      avgRec = avgRec + self.rec[i]
    avgPrec = avgPrec/len(prec)
    avgRec = avgRec/len(rec)

    self.F1 = 2*((avgPrec*avgRec)/(avgPrec+avgRec))
    #self.prec = avgPrec


    #self.F1 = 2*((avgPrec*avgRec)/(avgPrec+avgRec))

  def calculateReg(self, pred, facts):
    print("pred: ", pred, " \n\nfacts: ", facts)
    distance = float(0)
    distanceabs = float(0)
    total = 0
    for i in range(len(pred)):
      for j in range(len(pred[i])):
        if pd.isna(pred[i][j]):
          print('hello')
        else:
          distance = distance + ((float(facts[i][j]) - float(pred[i][j]))**float(2))
          distanceabs = distanceabs + abs((float(facts[i][j]) - float(pred[i][j])))
          self.error.append((float(facts[i][j]) - float(pred[i][j]))**float(2))
          total = total+1

    if len(facts) > 0:
      self.mse = distance/total
      self.mae = distanceabs/total
    else:
      self.mse = 0
      
      
      
      
 
