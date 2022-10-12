#-----------------------imports-------------------------
import pandas as pd
import numpy as np
from scipy import stats as st
import requests
import os
import random as random

class Model:

  def __init__(self):
    df = pd.DataFrame
    predictions = []
    labels = []

  def run(self, data, k_nn, k_cluster,value):
    
    #Bring in the 10 folds
    folds = data[1]
    thing3 = []
    thing4 = []
    
    for iterations in range(10):

      #Get an array to track which fold we are on
      all_folds = [0,1,2,3,4,5,6,7,8,9]
      all_folds.pop(iterations)

      #Get Testing data poping one out of the array at a time.
      testing_data = folds[iterations]
      if len(testing_data) == 0:
        break

      #Use the rest of the folds for training
      training_data = []
      for other_folds in all_folds:
        for contents in folds[other_folds]:
          training_data.append(contents)
          
      training_df_with_class = pd.DataFrame()
      testing_df_with_class = pd.DataFrame()

      #Turn these lists into dataframes
      for i in training_data:
        temp_df = pd.DataFrame(i)
        temp_df_T = temp_df.transpose()
        training_df_with_class = training_df_with_class.append(temp_df_T)
      for ii in testing_data:
        temp_df = pd.DataFrame(ii)
        temp_df_T = temp_df.transpose()
        testing_df_with_class = testing_df_with_class.append(temp_df_T)

      #Make the dataframe the correct shape and remove the classes.
      training_df = training_df_with_class.copy()
      testing_df = testing_df_with_class.copy()
      training_df = training_df.iloc[: , :-1]
      testing_df = testing_df.iloc[: , :-1]
      training_df = training_df.reset_index()
      testing_df = testing_df.reset_index()
      training_df_with_class = training_df_with_class.reset_index()
      testing_df_with_class = testing_df_with_class.reset_index()

      training_df = training_df.drop(columns=['index'])
      testing_df = testing_df.drop(columns=['index'])


      old_val_acc = 0
      val_acc = 1

      clicks = 1

      tracker = [0,0]

      #The following is our editing algorithm
      #We are going to repeat editting the training set until our accuraacy degrades
      while (tracker[clicks] > tracker[clicks-1] or clicks==1):

        tuning_data = data[0]   #grab the tuning data

        tuning_df_with_class = pd.DataFrame()

        for ii in tuning_data:
          temp_df = pd.DataFrame(ii)
          temp_df_T = temp_df.transpose()
          tuning_df_with_class = tuning_df_with_class.append(temp_df_T)

        #Bring in the tuning test so that we have something to edit against.
          
        tuning_df = tuning_df_with_class.copy()
        tuning_df = tuning_df.iloc[: , :-1]
        tuning_df = tuning_df.reset_index()
        tuning_df_with_class = tuning_df_with_class.reset_index()
        tuning_df = tuning_df.drop(columns=['index'])    #Conver the tuning data into a df

        decision = []
        df_matrix = pd.DataFrame(np.nan, index=range(len(tuning_df)), columns = range(len(training_df)))

        #Do KNN in order to classify the points so that we have an original validation accuracy to compare to
        
        for count1 in range(len(tuning_df)): #go throught the tuning df
          base = tuning_df.iloc[count1]
          for count2 in range(len(training_df)): #compare against training df
            dist1 = []
            for count3 in range(len(tuning_df.columns)):     
                dist1.append(((base[count3]) - (training_df.iloc[count2][count3]))**2)   #Find the distance   #1 error right here
            summation = sum(dist1)   #this seesm fine
            distance = np.sqrt(summation)   #this is fine for now as long as you are not getting a wierd error
            df_matrix.loc[count1, count2] = distance  

          #Get minimum index
          comparison_array = np.array(df_matrix.loc[count1])
          k = k_nn
          index = np.argpartition(comparison_array, k)
          reduced_idx = index[:k]
          reduced_list = reduced_idx.tolist()

          #Associate minimum index with class
          majority =[]
          for i in reduced_list:
            majority.append(training_df_with_class.iloc[i, -1])
          class_decision = st.mode(majority)
          decision.append(class_decision)

        counts = 0

        for i in range(len(decision)):
          if decision[i][0] == tuning_df_with_class.iloc[i, -1]:
            counts += 1
            
        val_acc = counts/len(tuning_df_with_class)   #calculate validation accuracy
        tracker.append(val_acc)  #append to tracker so that we know when to break out.


        #the above code was to give us a comparison as to when to stop

        ############################### Now we edit #######################################

        clicks = clicks + 1 #increment 1


        #Go through the training dataset and edit out as you go

        df_matrix = pd.DataFrame(np.nan, index=range(len(training_df)), columns = range(len(training_df)))

        decision = []

        counting = 0
        
        #Do Knn again

        for count1 in range(len(training_df)):
          base = training_df.iloc[count1-counting]
          for count2 in range(len(training_df)):
            if count1-counting != count2-counting:
              dist1 = []
              for count3 in range(len(training_df.columns)):
                  dist1.append(((base[count3]) - (training_df.iloc[count2][count3]))**2)   #compare everything in the training set.
              summation = sum(dist1)
              distance = np.sqrt(summation)
              df_matrix.loc[count1, count2] = distance 


          comparison_array = np.array(df_matrix.loc[count1])
          k = k_nn
          index = np.argpartition(comparison_array, k)
          reduced_idx = index[:k]

          reduced_list = reduced_idx.tolist()

          majority =[]
          for i in reduced_list:
            majority.append(training_df_with_class.iloc[i, -1])
          class_decision = st.mode(majority)


          #If the prediction and the acutal remove the misclassified point.
          if class_decision[0] != training_df_with_class.iloc[count1 - counting, -1]:
            training_df = training_df.drop(count1 - counting)
            training_df_with_class = training_df_with_class.drop(count1- counting)
            counting = counting + 1
            
            #Restrcutre the dataset in order to deal with the removed data
            if 'level_0' in training_df_with_class:
              training_df_with_class = training_df_with_class.drop(columns=['level_0'])
            training_df_with_class = training_df_with_class.reset_index()
            training_df = training_df.reset_index()
            training_df_with_class = training_df_with_class.drop(columns=['index'])
            training_df = training_df.drop(columns=['index'])



      df_matrix = pd.DataFrame(np.nan, index=range(len(training_df)), columns = range(len(training_df)))
      decision = []
      
      #KNN with the now reduced dataset versus each testing fold.
      for count1 in range(len(testing_df)):
        base = testing_df.iloc[count1]
        for count2 in range(len(training_df)):
          dist1 = []
          for count3 in range(len(testing_df.columns)):
              dist1.append(((base[count3]) - (training_df.iloc[count2][count3]))**2)
          summation = sum(dist1)
          distance = np.sqrt(summation)
          df_matrix.loc[count1, count2] = distance 

        #Find the minimum of each row and turn it into an array
        comparison_array = np.array(df_matrix.loc[count1])
        k = k_nn
        index = np.argpartition(comparison_array, k)
        reduced_idx = index[:k]
        reduced_list = reduced_idx.tolist()
        
        #Connect the class the index calculated above
        majority =[]
        for i in reduced_list:
          majority.append(training_df_with_class.iloc[i, -1])
        class_decision = st.mode(majority)
        decision.append(class_decision)

      counts = 0
      thing1 = []
      thing2 = []
      
      #Append the results from going through the training data once.

      for i in range(len(decision)):
        thing1.append(decision[i][0][0].item())
        thing2.append(testing_df_with_class.iloc[i, -1])
        if decision[i][0] == testing_df_with_class.iloc[i, -1]:
          counts += 1

      #If the output is a string as in dataset 4, convert it into a number for easier loss.
          
      for countss, i in enumerate(thing1):
        if i == 'D1': 
          thing1[countss] = 1
        if i == 'D2': 
          thing1[countss] = 2
        if i == 'D3': 
          thing1[countss] = 3
        if i == 'D4': 
          thing1[countss] = 4

      for countss, i in enumerate(thing2):
        if i == 'D1': 
          thing2[countss] = 1
        if i == 'D2': 
          thing2[countss] = 2
        if i == 'D3': 
          thing2[countss] = 3
        if i == 'D4': 
          thing2[countss] = 4

      thing3.append(thing1)
      thing4.append(thing2)

    self.labels = thing4
    self.predictions = thing3
