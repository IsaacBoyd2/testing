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
    prec = []
    rec = []
    error = []
    F1 = int()
  
  def calculate(self, value, classes, pred, facts):
    #makes a confusion Matrix that is the correct size
    confusionMat = np.zeros([len(classes),len(classes)])

    if value == 0:
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
          
        precision = truePos/(truePos+falsePos)
        recall = truePos/(truePos+falseNeg)

        self.prec.append(precision)
        self.rec.append(recall)

      #calculates the F1
      avgPrec = 0
      avgRec = 0
      for i in range(len(self.prec)):
        avgPrec = avgPrec + self.prec[i]
        avgRec = avgRec + self.rec[i]
      avgPrec = avgPrec/len(self.prec)
      avgRec = avgRec/len(self.rec)

      self.F1 = 2*((avgPrec*avgRec)/(avgPrec+avgRec))

    else:
      pass
