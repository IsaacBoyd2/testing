#import Pandas and Numpy

import pandas as pd
import numpy as np
import math
import random

#--------------------------------------
#Bring in the data

df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/c55c311d66e9dd04da9a6ee8627facdcc11e9d2a/DataProject1/glass.csv?raw=true')

#--------------------------------------


#Hyper Parameters

dev = 16 #standard deviation control
balance = True #Set to true to balance datasets

#--------------------------------------
#Preprocessing

bins = df['Glass Type'].unique()                                #Get all of the classes                       

training_size = math.ceil(len(df)*4/5)                          #Split the data 80/20 by index
random_list = random.sample(range(len(df)), len(df)) 


#Create a list of training and testing data
training_list = random_list[0:training_size]
testing_list = random_list[training_size:len(df)]

#Create a testing and training dataframe from the lists
training_df =  df.iloc[:, 1:len(df.columns)]
testing_df_with_labels = df.iloc[testing_list]
testing_df = testing_df_with_labels.iloc[: , 1:-1]

#category_df = training_df[training_df['Glass Type'] == 1]
#print(category_df.loc[random.randint(0, len(category_df)-1)])

def balancing(training_df):

  max_list = []

  for i in bins:
    category_df = training_df[training_df['Glass Type'] == i]
    max_list.append(len(category_df))
  
  for i in bins:
    category_df = training_df[training_df['Glass Type'] == i]
    while len(category_df) < max(max_list):
      #df2 = df2.append(df1.iloc[x])
      training_df = training_df.append(category_df.iloc[random.randint(0, len(category_df)-1)])
      category_df = training_df[training_df['Glass Type'] == i]
      #print(len(category_df))

  return training_df

if balance == True:
  training_df = balancing(training_df)

#--------------------------------------


#Model/Algorithm

'''
Breif Description of model:
Model takes in a training and testing dataframe. Each entry into the training dataframe is
an unknown entry with several attributes. These attributes are compared to those in the training
set and a estimation is made based on a range as to which class shares the most attributes with
the input data.
'''

results = []

def model(training_df, testing_df):

  for lines in range(len(testing_df)):                                         #For every testing datum
    row = testing_df.iloc[lines]                              
    C_x = []
    for i in bins:                                                             #For every class (so in other words every testing input will be compared to every class)
      F_a_c_list = []                                                 
      category_df = training_df[training_df['Glass Type'] == i]  #This splits the data into class spesific dataframes       
      for count,j in enumerate(row):                                                                                  
        y = 0
        for k in category_df.iloc[:, count][0:len(category_df)]:               #For every attribute in the inputs
          sd = np.std(category_df.iloc[:, count][0:len(category_df)])          #Compare every attribute to all other attributes in the class
          if k < (j+(sd/16)) and k>(j-(sd/16)):                                #If the attribute is close to another attribute (withing a fration of an standard deviation) add one
            y = y + 1
          
        numerator = y + 1
        denominator = len(category_df)+len(testing_df.columns)-1               #The following computes for the similarity based on the algotrim
                                                                               #C(x) = Q(C = ci) × d∏j=1 F (Aj = ak, C = ci)
        F_a_c = numerator/denominator
        F_a_c_list.append(F_a_c)

      C_x.append(np.prod(F_a_c_list)*(len(category_df)/len(df)))

    results.append(bins[C_x.index(max(C_x))])                                  #Append the results for future comparisons
           


model(training_df,testing_df)


#--------------------------------------

#Analysis (WIP)

correct = 0
for count,i in enumerate(results):
  if testing_df_with_labels['Glass Type'].iloc[count] == i:
    correct += 1

accuracy = correct/len(results)
print(accuracy)

print(results)
print(testing_df_with_labels['Glass Type'])

