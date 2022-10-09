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
  def __init__ (self):
    F1 = []
    precision = []
    recall = []
    error = []
  
  def calculate(self, value, classes, pred, facts):
    confusionMat = np.zeros([len(classes),len(classes)])
    if value == 0:
      #Creates a Confusion Matrix that will be used for precision, recall, and F1 calculations.
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

      print("\n\n\n\nConfusionMat: ", confusionMat)

    else:
      pass
