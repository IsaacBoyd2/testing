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

classInputArray = [['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/code/preprocessing.py?raw=true','preprocessing.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/code/loss.py?raw=true', 'loss.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/code/mlp.py?raw=true', 'mlp.py']]


for i in range(len(classInputArray)):
  with open(classInputArray[i][1], 'w') as f:
    r = requests.get(classInputArray[i][0])
    f.write(r.text)

import preprocessing as pp
import loss as lss
import mlp as MLP

#------------------------Main--------------------------

def main():

  #-----Hyper_parameters-------#

  hiddenArray = [2,3]

  #----------------------------#

  #runs preProcessing
  preProcess = pp.Preprocessing()
  preProcess.process()
  preProcess.fold()
  data = preProcess.folds

  #classification
  if preProcess.value == 0:
    preProcess.oneHot()
    modeling = MLP.Model()
    modeling.run(len(preProcess.df.iloc[0]),hiddenArray,len(preProcess.classes))
    for i in range(len(preProcess.folds[0:8])): 
      modeling.forwardProp(preProcess.df.values[i,0:-1].astype('float'),preProcess.value)
      print("modeling values: ", modeling.values)
      #add back prop here 

  #regression
  else:
    modeling = MLP.Model()
    modeling.run(len(preProcess.df.iloc[0]),hiddenArray,1)
    for i in range(len(preProcess.folds[0:8])):
      modeling.forwardProp(preProcess.df.values[i,0:-1].astype('float'),preProcess.value)
      #add back prop here
  
 



#calls main
main()

#makes sure that the python classes are taken out after execution
# for i in range(len(classInputArray)):
  # fileName = classInputArray[i][1]
  # os.remove(fileName)
