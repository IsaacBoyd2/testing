#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: Naive Bayes
##Completed: 9-29-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd

#------------------------Class--------------------------
class Preprocessing:
  def __init__ (self):
    pass

  #method that allows the user to chose the dataset, and preprocesses that said data set.
  def process(self):
    #gets user input 
    DataNumber = input("Enter a Number(1-6) for data selection")

    if DataNumber == '1':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/breast-cancer-wisconsin.csv?raw=true')
      print("Using Breast Cancer data (Classification)")
      for i in range(len(df)):
        if i < len(df):
          for j in range(len(df.columns)):
            if df.iloc[i,j] == "?":
              #takes out the row with missing values
              df = df.drop(df.index[i])
              i = i - 1

    elif DataNumber == '2':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/glass.csv?raw=true')
      print("Using Glass data (Classification)")
    elif DataNumber == '3':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/soybean-small.csv?raw=true')
      print("Using Soybean data (Classification)")
    elif DataNumber == '4':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/abalone.csv?raw=true")
      print("Using Abalone data (Regression)")
    elif DataNumber == '5':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/machine.csv?raw=true")
      print("Using Machine data (Regression)")
    elif DataNumber == '6':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/forestfires.csv?raw=true")
      print("Using Forest Fires data (Regression)")
    else:
      print("That is not a valid value for picking the data set.")
      
preProcess = Preprocessing() 
preProcess.process()
