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

  def run(self, data, k_nn, sigma, epsilon):

    folds = data[1]
    thing3 = []
    thing4 = []
    for iterations in range(10):

      #Array for indexing folds
      all_folds = [0,1,2,3,4,5,6,7,8,9]
      all_folds.pop(iterations)

      #Split into training and testing
      testing_data = folds[iterations]
      if len(testing_data) == 0:
        break
      training_data = []
      for other_folds in all_folds:
        for contents in folds[other_folds]:
          training_data.append(contents)

      training_df_with_class = pd.DataFrame()
      testing_df_with_class = pd.DataFrame()

      #Convert into dataframe
      for i in training_data:
        temp_df = pd.DataFrame(i)
        temp_df_T = temp_df.transpose()
        training_df_with_class = training_df_with_class.append(temp_df_T)
      for ii in testing_data:
        temp_df = pd.DataFrame(ii)
        temp_df_T = temp_df.transpose()
        testing_df_with_class = testing_df_with_class.append(temp_df_T)
        
      #Reshape to fit needs
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

      #Break out function
      while (tracker[clicks] > tracker[clicks-1] or clicks==1):
        
        #Set up tuning dataframe
        tuning_data = data[0]   #grab the tuning data
        tuning_df_with_class = pd.DataFrame()

        for ii in tuning_data:
          temp_df = pd.DataFrame(ii)
          temp_df_T = temp_df.transpose()
          tuning_df_with_class = tuning_df_with_class.append(temp_df_T)

        tuning_df = tuning_df_with_class.copy()
        tuning_df = tuning_df.iloc[: , :-1]
        tuning_df = tuning_df.reset_index()
        tuning_df_with_class = tuning_df_with_class.reset_index()
        tuning_df = tuning_df.drop(columns=['index'])    #Conver the tuning data into a df

        decision = []

        #Do an initial KNN so that we have something to compare against for our breakout.
        df_matrix = pd.DataFrame(np.nan, index=range(len(tuning_df)), columns = range(len(training_df)))

        for count1 in range(len(tuning_df)): #go throught the tuning df
          base = tuning_df.iloc[count1]
          for count2 in range(len(training_df)): #compare against training df
            dist1 = []
            for count3 in range(len(tuning_df.columns)):     
                dist1.append(((base[count3]) - (training_df.iloc[count2][count3]))**2)   #Find the distance   #1 error right here
            summation = sum(dist1)   
            distance = np.sqrt(summation)   
            df_matrix.loc[count1, count2] = distance  

          #Put everything into a comparison array to get the inexes of the knn..
          comparison_array = np.array(df_matrix.loc[count1])
          k = k_nn
          index = np.argpartition(comparison_array, k)
          reduced_idx = index[:k]
          reduced_list = reduced_idx.tolist()

          #Associate index with class values
          majority =[]
          for i in reduced_list:
            majority.append(training_df_with_class.iloc[i, -1])
          class_decision = st.mode(majority)
          decision.append(class_decision)

        counts = 0
        
        #Test if decion value is within epsilon criterion.
        for i in range(len(decision)):
          if abs(tuning_df_with_class.iloc[i, -1]-decision[i][0][0])/tuning_df_with_class.iloc[i, -1] < epsilon:
            counts += 1

        val_acc = counts/len(tuning_df_with_class)   #calculate validation accuracy
        tracker.append(val_acc)
        #the above code was to give us a comparison as to when to stop

        ############################### Now we edit #######################################

        clicks = clicks + 1 

        #Go through the training dataset and edit out as you go
        df_matrix = pd.DataFrame(np.nan, index=range(len(training_df)), columns = range(len(training_df)))

        decision = []

        counting = 0

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

          #Find min idxs
          comparison_array = np.array(df_matrix.loc[count1])
          k = k_nn
          index = np.argpartition(comparison_array, k)
          reduced_idx = index[:k]
          reduced_list = reduced_idx.tolist()

          #associate with class values
          majority =[]
          #print(counting)
          #print(training_df_with_class)
          for i in reduced_list:

            #print(i-counting)

            majority.append(training_df_with_class.iloc[i, -1])   #removed counting. Maybe now that drop is true we do not need to reset the counting or something?


          class_decision = st.mode(majority)

          #print(len(training_df_with_class))

          #print(abs(training_df_with_class.iloc[count1-counting, -1]-class_decision[0])/training_df_with_class.iloc[count1-counting, -1])

          if abs(training_df_with_class.iloc[count1-counting, -1]-class_decision[0])/training_df_with_class.iloc[count1-counting, -1] > epsilon: #second epsilon should go here but have the same criterion but obviously we are editing if it is wrong so this needs to be greater than
            #print(training_df[0:10])
            training_df = training_df.drop(count1 - counting)
            training_df_with_class = training_df_with_class.drop(count1- counting)
            counting = counting + 1

            print(len(training_df_with_class))

            #print(training_df_with_class)
            if 'level_0' in training_df_with_class:
              training_df_with_class = training_df_with_class.drop(columns=['level_0'])

            training_df_with_class = training_df_with_class.reset_index()
            training_df = training_df.reset_index()

            training_df_with_class = training_df_with_class.drop(columns=['index'])
            training_df = training_df.drop(columns=['index'])
            #training_df_with_class = training_df_with_class.drop(columns=['level_0'])
            #raining_df = training_df.drop(columns=['level_0'])
        #sys.exit()




      df_matrix = pd.DataFrame(np.nan, index=range(len(testing_df)), columns = range(len(training_df)))

      decision = []
      thing1 = []
      thing2 = []

      for count1 in range(len(testing_df)):
        base = testing_df.iloc[count1]
        for count2 in range(len(training_df)):
          dist1 = []
          for count3 in range(len(testing_df.columns)):
              dist1.append(((base[count3]) - (training_df.iloc[count2][count3]))**2)


          summation = sum(dist1)
          distance = np.sqrt(summation)

          df_matrix.loc[count1, count2] = distance 

        comparison_array = np.array(df_matrix.loc[count1])
        k = k_nn
        index = np.argpartition(comparison_array, k)
        reduced_idx = index[:k]

        reduced_list = reduced_idx.tolist()



        desicion_list = []
        for i in reduced_list:

              #last_column = df.iloc[: , -1]
              #desicion_list.append(training_df_with_class['Rings'][i])
          desicion_list.append(training_df_with_class.iloc[i, -1])        

        denominator = []
        numerator = []


        for i in desicion_list:
          weight_holder = []
          for ii in desicion_list:
            if i != ii:
              w_i_part = 2.72**((-(i-ii)**2)/sigma**2)

              #print(w_i)
              weight_holder.append(w_i_part)
          w_i = sum(weight_holder)

          #print(w_i)
          denominator.append(w_i)
          y_i_w_i = w_i * i
          numerator.append(y_i_w_i) 

        f = sum(numerator)/sum(denominator)
        #print("numerator: ", numerator)
        #print("denominator: ", denominator)

        thing1.append(f)
        thing2.append(testing_df_with_class.iloc[count1, -1])



      thing3.append(thing1)
      thing4.append(thing2)



    self.labels = thing4
    self.predictions = thing3

    print(self.labels)
    print(self.predictions)
