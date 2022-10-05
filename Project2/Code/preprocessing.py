# -*- coding: utf-8 -*-
"""Preprocessing

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1255s_i_-ARo5T5W7qRQV4RlMP4LL2i-S
"""

#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: Preprocessing
##Completed: 9-29-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import math
import random

#------------------------Class--------------------------
class Preprocessing:
  def __init__ (self):
    df = pd.DataFrame
    trainingDf = pd.DataFrame
    testingDf = pd.DataFrame

  #method that allows the user to chose the dataset, and preprocesses that said data set.
  def process(self):
    #gets user input 
    DataNumber = input("Enter a Number(1-6) for data selection: ")

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
      self.df = df

    elif DataNumber == '2':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/glass.csv?raw=true')
      print("Using Glass data (Classification)")
      self.df = df

    elif DataNumber == '3':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/soybean-small.csv?raw=true')
      print("Using Soybean data (Classification)")
      self.df = df

    elif DataNumber == '4':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/abalone.csv?raw=true")
      print("Using Abalone data (Regression)")

      oneEncodeList = []

      for i in range(len(df)):
        print("loop " + str(i))
        if df.iloc[0,i] == 'M':
          oneEncodeList.append([1,0,0])
        elif df.iloc[0,i] == 'F':
          oneEncodeList.append([0,1,0])
        else:
          oneEncodeList.append([0,0,1])

      adding = pd.DataFrame(oneEncodeList, columns =['Male', 'Female', 'Other'])
      df.join(adding)
      df.drop('Sex')
      print(adding)
      print(df)
      self.df = df

    elif DataNumber == '5':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/machine.csv?raw=true")
      print("Using Machine data (Regression)")
      self.df = df

    elif DataNumber == '6':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/forestfires.csv?raw=true")
      print("Using Forest Fires data (Regression)")
      self.df = df

    else:
      print("That is not a valid value for picking the data set.")

  def fold(self):
    #takes input for the number of folds (mostly 10 in this case)
    foldNumber = int(input("How many folds?: "))

    randomList = random.sample(range(len(self.df)), len(self.df))
    testingList = []
    trainingList = []
    testingSize = len(self.df) - math.ceil(len(self.df)*(foldNumber-1)/(foldNumber))

    testingList = randomList[testingSize:testingSize]
    trainingList = randomList.copy()

    for i in testingList:
      trainingList.remove(i)

    self.trainingDf = self.df.iloc[trainingList]
    self.testingDf = self.df.iloc[testingList]