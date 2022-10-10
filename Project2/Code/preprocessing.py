#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: Preprocessing
##Completed: 10-6-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import math
import random
import numpy as np

#------------------------Class--------------------------
class Preprocessing:

  #initialization
  def __init__ (self):
    self.df = pd.DataFrame
    self.folds = []
    self.tuning = []
    self.value = int()

  #method that allows the user to chose the dataset, and preprocesses that said data set.
  def process(self):
    #gets user input 
    DataNumber = input("Enter a Number(1-6) for data selection: ")

    if DataNumber == '1':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/breast-cancer-wisconsin.csv?raw=true')
      print("Using Breast Cancer data (Classification)")

      #removes the rows with question marks
      df = df[~(df=='?').any(axis=1)].reset_index(drop=True)

      for i in range(len(df)):
        for j in range(len(df.iloc[0])):
          df.iloc[i][j] = int(df.iloc[i][j])

      self.value = 0
      self.df = df

    elif DataNumber == '2':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/glass.csv?raw=true')

      #for j in df.columns:
      #  df[j] = (df[j] - df[j].min()) / (df[j].max() - df[j].min())

      print("Using Glass data (Classification)")
      self.value = 0
      self.df = df

    elif DataNumber == '3':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/soybean-small.csv?raw=true')
      print("Using Soybean data (Classification)")
      self.value = 0
      self.df = df

    elif DataNumber == '4':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/abalone.csv?raw=true")
      print("Using Abalone data (Regression)")

      oneEncodeList = []

      for i in range(len(df)):
        if df.iloc[i,0] == 'M':
          oneEncodeList.append([1,0,0])
        elif df.iloc[i,0] == 'F':
          oneEncodeList.append([0,1,0])
        else:
          oneEncodeList.append([0,0,1])

      adding = pd.DataFrame(oneEncodeList, columns =['Male', 'Female', 'Other'])
      df = df.join(adding)
      df.drop(columns=["Sex"], inplace = True)
      df = df.sort_values(by=['Rings'])

      classColumn = df.pop("Rings")

      df.insert(len(df.columns), "Rings", classColumn)
      df = df.reset_index()

      self.value = 1
      self.df = df

    elif DataNumber == '5':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/machine.csv?raw=true")
      print("Using Machine data (Regression)")

      df.drop(columns=["VendorName"],  inplace = True)
      df.drop(columns=["ModelName"],  inplace = True)
      df = df.sort_values(by=['PRP'])

      self.value = 1
      self.df = df

    elif DataNumber == '6':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/forestfires.csv?raw=true")
      print("Using Forest Fires data (Regression)")

      months = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
      days = {'sun':1,'mon':2,'tue':3,'wed':4,'thu':5,'fri':6,'sat':7}

      for i in range(len(df)):
        df = df.replace({"month":months})
        df = df.replace({"day":days})
        df = df.sort_values(by=['area'])

      self.value = 1
      self.df = df

    else:
      print("That is not a valid value for picking the data set.")

    #return self.df

  #function used to make the tuning fold, and 'foldNumber' amount of folds
  def fold(self):
    #number of folds (mostly 10 in this case)
    foldNumber = 10
    tuningList = []
    fold = []
    foldTotal = []
    randomList = random.sample(range(len(self.df)), len(self.df))

    #folds depending on categorical or regression data

    #categorical
    if self.value == 0:
      classArray = self.df['Class']
      classes = self.df['Class'].unique()
      classSize = np.zeros((len(classes),1))
      for i in range(len(self.df)):
        for j in range(len(classes)):
          if classArray.iloc[i] == classes[j]:
            classSize[j] = classSize[j] + 1

      total = len(self.df)
      classPercent = np.zeros((len(classes),1))

      for i in range(len(classSize)):
        classPercent[i] = classSize[i]/total

      total = total*.1
      classAmount = np.zeros((len(classes),1))

      #gets the amount that we need from each class for the tuning array
      for i in range(len(classSize)):
        classAmount[i] = math.ceil(classPercent[i] * total)

      dfHolder = self.df

      #used to get the randomized tuning list
      for i in range(len(classAmount)):
        for j in range(int(classAmount[i][0])):
          for k in range(len(randomList)):
            if classArray.iloc[randomList[k]] == classes[i]:
              tuningList.append(dfHolder.iloc[randomList[k]])
              dfHolder = dfHolder.drop(randomList[k])
              dfHolder = dfHolder.reset_index(drop=True)
              classArray = classArray.drop(randomList[k])
              classArray = classArray.reset_index(drop=True)
              for l in range(len(randomList)):
                if randomList[l] > k:
                  randomList[l] = randomList[l] - 1
              randomList.remove(k)
              break

      self.tuning = tuningList

      total = len(dfHolder)

      classSize = np.zeros((len(classes),1))

      for i in range(len(dfHolder)):
        for j in range(len(classes)):
          if classArray.iloc[i] == classes[j]:
            classSize[j] = classSize[j] + 1

      for i in range(len(classSize)):
        classPercent[i] = classSize[i]/total

      #gets the new amount of each class that we need for each fold
      for i in range(len(classPercent)):
        classAmount[i] = math.ceil(classPercent[i] * (len(dfHolder)/foldNumber))

      #used to get the 'foldNumber' amount of folds
      for m in range(foldNumber):
        fold = []
        #loops through the amount of classes
        for i in range(len(classAmount)):
          #loops for as many of each class we need
          for j in range(int(classAmount[i][0])):
            #loops through the randomList in order to find classes that are needed
            for k in range(len(randomList)):
              if classArray.iloc[randomList[k]] == classes[i]:
                fold.append(dfHolder.iloc[randomList[k]])
                dfHolder = dfHolder.drop(randomList[k])
                dfHolder = dfHolder.reset_index(drop=True)
                classArray = classArray.drop(randomList[k])
                classArray = classArray.reset_index(drop=True)
                #changes all of the indexes greater than the one that we are taking out
                for l in range(len(randomList)):
                  if randomList[l] > k:
                    randomList[l] = randomList[l] - 1
                randomList.remove(k)
                break
        foldTotal.append(fold)

      self.folds = foldTotal


    #regression
    else:
      #sets values for looping through the data
      index = 0
      total = len(self.df)
      percentage = total*.1
      percentage = math.ceil(len(self.df)/percentage)
      foldAmount = math.ceil(total*.1)

      dfFolds = []
      lastIndex = 0

      #loops through the amount of folds, then make a 'cut' at each 'class'
      for i in range(foldNumber):
        if lastIndex < len(self.df):
          dfFolds.append(self.df.iloc[(lastIndex):lastIndex+foldAmount])
          lastIndex = lastIndex + foldAmount
        else:
          dfFolds.append(self.df.iloc[lastIndex:len(self.df)])

      #shuffles the classes
      for i in range(len(dfFolds)):
        dfFolds[i] = dfFolds[i].sample(frac=1).reset_index()
          
      #loops to create the tuningList
      for i in range(len(dfFolds)):
        for j in range(math.floor(foldAmount/foldNumber)):
          tuningList.append(dfFolds[i].iloc[j])
          dfFolds[i] = dfFolds[i].drop(j)
        dfFolds[i].reset_index(drop=True)

      self.tuning = tuningList

      #used to get a new total and reindex all of the classes
      total = 0
      for i in range(len(dfFolds)):
        total = total + len(dfFolds[i])
        dfFolds[i] = dfFolds[i].reset_index(drop=True)
      foldAmount = math.ceil(total*.1)

      foldTotal = []
      #loops for the amount of folds
      for k in range(foldNumber):
        fold = []
        #goes through each class in order to stratisfy the data
        for i in range(len(dfFolds)):
          #gives each fold the correct amount of points
          for j in range(math.ceil(foldAmount/foldNumber)):
            if j < len(dfFolds[i]):
              fold.append(dfFolds[i].iloc[j])
              dfFolds[i] = dfFolds[i].drop(j)
          dfFolds[i] = dfFolds[i].reset_index(drop=True)
        
        foldTotal.append(fold)

      self.folds = foldTotal

#preProcess = Preprocessing()
#preProcess.process()
#preProcess.fold()
