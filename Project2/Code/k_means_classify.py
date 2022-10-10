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

  def run(self, data, k_nn):

    #preProcess = pp.Preprocessing()
    #preProcess.process()
    #preProcess.fold()
    #data = [preProcess.tuning, preProcess.folds]

  
    folds = data[1]
    thing3 = []
    thing4 = []

    for iterations in range(10):

      print(iterations*10, '%')

      all_folds = [0,1,2,3,4,5,6,7,8,9]

      all_folds.pop(iterations)

      testing_data = folds[iterations]

      if len(testing_data) == 0:
        break

      training_data = []
      for other_folds in all_folds:
        for contents in folds[other_folds]:
          training_data.append(contents)

      training_df_with_class = pd.DataFrame()

      testing_df_with_class = pd.DataFrame()
      
      for i in training_data:
        temp_df = pd.DataFrame(i)
        temp_df_T = temp_df.transpose()
        training_df_with_class = training_df_with_class.append(temp_df_T)
      for ii in testing_data:
        temp_df = pd.DataFrame(ii)
        temp_df_T = temp_df.transpose()
        testing_df_with_class = testing_df_with_class.append(temp_df_T)

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

      clicks = 0

      Initial_weights = [0] * len(training_df)

      new_weights = [1] * len(training_df)


      while (Initial_weights != new_weights):

        Initial_weights = new_weights.copy() 

        

        if clicks == 0:

          weights_matrix = pd.DataFrame(np.nan, index=range(k_nn), columns = range(len(training_df.columns)))


          randomList = random.sample(range(len(training_df)), k_nn)

          for counters,i in enumerate(randomList):
            weights_matrix.iloc[counters] = training_df.iloc[i]

        clicks = clicks + 1

        df_matrix = pd.DataFrame(np.nan, index=range(len(weights_matrix)), columns = range(len(training_df)))

        Centroid_holder = []

        for i in range(k_nn):
            Centroid_holder.append([])

        for count1 in range(len(weights_matrix)):
          base = weights_matrix.iloc[count1]
          for count2 in range(len(training_df)):
            dist1 = []
            for count3 in range(len(weights_matrix.columns)):
              dist1.append((float(base[count3]) - float(training_df.iloc[count2][count3]))**2)
        
            summation = sum(dist1)
            distance = np.sqrt(summation)

            df_matrix.iloc[count1, count2] = distance 

        for i in df_matrix.columns:
          new_weights[i] = df_matrix.iloc[:,i].idxmin()
          Centroid_holder[int(df_matrix.iloc[:,i].idxmin())].append(i)

        for i in range(len(Centroid_holder)):
          df_holder = pd.DataFrame(np.nan, index=range(len(Centroid_holder[i])), columns = range(len(training_df.columns)))
          for the_count,ii in enumerate(Centroid_holder[i]):
            df_holder.iloc[the_count] =  training_df.iloc[ii]

          df2_holder = pd.DataFrame(np.nan, index=range(1), columns = range(len(training_df.columns)))

          for iii in range(len(df_holder.columns)):
            df2_holder[iii] = df_holder[iii].mean()

          p = df2_holder.to_numpy()

          if np.isnan(p).all() == False:
            weights_matrix.iloc[i] = df2_holder



      weights_matrix_labels = weights_matrix.copy()

      weights_matrix_labels['Class'] = 1

      df_matrix = pd.DataFrame(np.nan, index=range(len(weights_matrix)), columns = range(len(training_df)))

      decision = []

      for count1 in range(len(weights_matrix)):
        base = weights_matrix.iloc[count1]
        for count2 in range(len(training_df)):
          dist1 = []
          for count3 in range(len(weights_matrix.columns)):
            dist1.append((float(base[count3]) - float(training_df.iloc[count2][count3]))**2)
        
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

        weights_matrix_labels['Class'][count1] = class_decision[0]
      
      df_matrix = pd.DataFrame(np.nan, index=range(len(testing_df)), columns = range(len(weights_matrix)))

      decision = []

      for count1 in range(len(testing_df)):
        base = testing_df.iloc[count1]
        for count2 in range(len(weights_matrix)):
          dist1 = []
          for count3 in range(len(testing_df.columns)):
            dist1.append((float(base[count3]) - float(training_df.iloc[count2][count3]))**2)
        
          summation = sum(dist1)
          distance = np.sqrt(summation)

          df_matrix.loc[count1, count2] = distance 

        comparison_array = np.array(df_matrix.loc[count1])
        k = k_nn

        index = np.argpartition(comparison_array, k-1)
        
        reduced_idx = index[:k]

        reduced_list = reduced_idx.tolist()

        print(reduced_list)

        #print(len(weights_matrix))

        majority =[]
        for i in reduced_list:
          majority.append(weights_matrix.iloc[i, -1])

        class_decision = st.mode(majority)
        
        decision.append(class_decision)

      counts = 0

      thing1 = []
      thing2 = []

      print(decision)

      for i in range(len(decision)):
        thing1.append(decision[i][0][0])
        thing2.append(testing_df_with_class.iloc[i, -1])

      thing3.append(thing1)
      thing4.append(thing2)

    self.labels = thing4
    self.predictions = thing3

    print(self.labels)
    print(self.predictions)
