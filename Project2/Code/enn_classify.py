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

  def run(self, data, k_nn, k_cluster):

    folds = data[1]
    thing3 = []
    thing4 = []

    for iterations in range(10):

      print(iterations)

      

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


      print(training_df)
      print(testing_df)

      
      old_val_acc = 0
      val_acc = 1

      clicks = 1

      tracker = [0,0]


      while (tracker[clicks] > tracker[clicks-1] or clicks==1):

        print(tracker)
        print(len(training_df))

        #################################### val test ################################################

        tuning_data = data[0]

        tuning_df_with_class = pd.DataFrame()

        for ii in tuning_data:
          temp_df = pd.DataFrame(ii)
          temp_df_T = temp_df.transpose()
          tuning_df_with_class = tuning_df_with_class.append(temp_df_T)


        tuning_df = tuning_df_with_class.copy()
        tuning_df = tuning_df.iloc[: , :-1]
        tuning_df = tuning_df.reset_index()
        tuning_df_with_class = tuning_df_with_class.reset_index()

        tuning_df = tuning_df.drop(columns=['index'])

        decision = []

        df_matrix = pd.DataFrame(np.nan, index=range(len(tuning_df)), columns = range(len(training_df)))

        for count1 in range(len(tuning_df)):
          base = tuning_df.iloc[count1]
          for count2 in range(len(training_df)):
            dist1 = []
            for count3 in range(len(tuning_df.columns)):
              if int(base[count3]) > 10**5:
                base[count3] = 0

              if value != 6:
                dist1.append((base[count3]) - (training_df.iloc[count2][count3])**2)
              else:
                dist1.append(((int(base[count3])) - int(training_df.iloc[count2][count3]))**2)
          
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

          decision.append(class_decision)

        counts = 0

        for i in range(len(decision)):
          if decision[i][0] == tuning_df_with_class.iloc[i, -1]:
            counts += 1

        print(len(decision))
        print(len(tuning_df_with_class))
        print(counts)


        val_acc = counts/len(tuning_df_with_class)

        tracker.append(val_acc)
        #print(tracker)
        print("Validation_accuracy",val_acc)








        ###########################################################################################

        clicks = clicks + 1
        
        df_matrix = pd.DataFrame(np.nan, index=range(len(training_df)), columns = range(len(training_df)))


        decision = []
        
        counting = 0

        for count1 in range(len(training_df)):
          base = training_df.iloc[count1-counting]
          for count2 in range(len(training_df)):
            if count1-counting != count2-counting:
              
              
              dist1 = []
              for count3 in range(len(training_df.columns)):
                if int(base[count3]) > 10**5:
                base[count3] = 0

                if value != 6:
                  dist1.append((base[count3]) - (training_df.iloc[count2][count3])**2)
                else:
                  dist1.append(((int(base[count3])) - int(training_df.iloc[count2][count3]))**2)
            
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
            majority.append(training_df_with_class.iloc[i-counting, -1])


          class_decision = st.mode(majority)



          if class_decision[0] != training_df_with_class.iloc[count1 - counting, -1]:
            #print(training_df[0:10])
            training_df = training_df.drop(count1 - counting)
            training_df_with_class = training_df_with_class.drop(count1- counting)
            counting = counting + 1

            training_df_with_class = training_df_with_class.reset_index()
            training_df = training_df.reset_index()

            training_df_with_class = training_df_with_class.drop(columns=['index'])
            training_df = training_df.drop(columns=['index'])

      


      df_matrix = pd.DataFrame(np.nan, index=range(len(training_df)), columns = range(len(training_df)))

      decision = []

      for count1 in range(len(testing_df)):
        base = testing_df.iloc[count1]
        for count2 in range(len(training_df)):
          dist1 = []
          for count3 in range(len(testing_df.columns)):
            if int(base[count3]) > 10**5:
                base[count3] = 0

            if value != 6:
              dist1.append((base[count3]) - (training_df.iloc[count2][count3])**2)
            else:
              dist1.append(((int(base[count3])) - int(training_df.iloc[count2][count3]))**2)
        
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

        decision.append(class_decision)

      counts = 0
      thing1 = []
      thing2 = []

      for i in range(len(decision)):
        thing1.append(decision[i][0][0].item())
        thing2.append(testing_df_with_class.iloc[i, -1])
        if decision[i][0] == testing_df_with_class.iloc[i, -1]:
          counts += 1
      
      thing3.append(thing1)

      print(thing3)
      thing4.append(thing2)

    print(thing4)
    print(thing3)

    self.labels = thing4
    self.predictions = thing3

    print(self.labels)
    print(self.predictions)
