#-----------------------imports-------------------------
import pandas as pd
import numpy as np
from scipy import stats as st

class Model:

  def __init__(self):
    df = pd.DataFrame
    predictions = []
    labels = []

  def run(self, data, k_nn):

    #Hyper parameters
    #k_nn = 3

    
    #first we want to talk data[1] and from it we can seperate it into its folds
    
    folds = data[1]

    #now we are going to run each fold and hold out one.

    accuracies = []
    thing3 = []
    thing4 =[]


    for iterations in range(10):
      all_folds = [0,1,2,3,4,5,6,7,8,9]
      all_folds.pop(iterations)

      testing_data = folds[iterations]

      training_data = []
      for other_folds in all_folds:
        for contents in folds[other_folds]:
          training_data.append(contents)

      training_df_with_class = pd.DataFrame()

      testing_df_with_class = pd.DataFrame()
      
      for i in training_data:
        temp_df = pd.DataFrame(i)
        temp_df_T = temp_df.transpose()
        #print(temp_df_T)
        training_df_with_class = training_df_with_class.append(temp_df_T)
        #print(training_df_with_class)
        #training_df_with_class = training_df_with_class.drop(columns=['ID Number'])
        #print(training_df_with_class)

      for ii in testing_data:
        temp_df = pd.DataFrame(ii)
        temp_df_T = temp_df.transpose()
        #print(temp_df_T)
        testing_df_with_class = testing_df_with_class.append(temp_df_T)
        #testing_df_with_class = testing_df_with_class.drop(columns=['ID Number'])
        #print(testing_df)

      training_df = training_df_with_class.copy()
      testing_df = testing_df_with_class.copy()

      training_df = training_df.iloc[: , :-1]
      testing_df = testing_df.iloc[: , :-1]
      #training_df = training_df.drop(columns=['Rings'])
      #testing_df = testing_df.drop(columns=['Rings'])
      training_df = training_df.reset_index()
      testing_df = testing_df.reset_index()
      training_df_with_class = training_df_with_class.reset_index()
      testing_df_with_class = testing_df_with_class.reset_index()
      training_df = training_df.drop(columns=['index'])
      testing_df = testing_df.drop(columns=['index'])
      #training_df_with_class = training_df_with_class.drop(columns=['ID Number'])
      
      #print(training_df)
      #print(training_df_with_class)
        

      #for count2, ii in enumerate(testing_data):
      # testing_df.iloc[count2] = ii

      #print(folds)
      #print(training_df)
      #print(testing_df)
        #for allof in all_folds:
        #  for peices in 



      #all_folds = range(10)

      #training = folds[1]

      #df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/glass.csv?raw=true')

      df_matrix = pd.DataFrame(np.nan, index=range(len(training_df)), columns = range(len(training_df)))


      #df = df.drop(0)
      #df = df.reset_index()
      #df2 = df.iloc[: , 2:-1]

      decision = []

      for count1 in range(len(testing_df)):
        base = testing_df.iloc[count1]
        #rint('hello')
        #print(base)
        #print('hello')
        for count2 in range(len(training_df)):
          dist1 = []
          for count3 in range(len(testing_df.columns)):
            dist1.append((base[count3] - training_df.iloc[count2][count3])**2)
        
          summation = sum(dist1)
          distance = np.sqrt(summation)

          df_matrix.loc[count1, count2] = distance 

        comparison_array = np.array(df_matrix.loc[count1])
        k = k_nn
        index = np.argpartition(comparison_array, k)
        reduced_idx = index[:k]

        reduced_list = reduced_idx.tolist()

        #print(reduced_list)
        #print(count1)


        #reduced_list.remove(count1)


        desicion_list = []
        for i in reduced_list:

          #last_column = df.iloc[: , -1]
          #desicion_list.append(training_df_with_class['Rings'][i])
          desicion_list.append(training_df_with_class.iloc[i, -1])

        #majority =[]
        #for i in reduced_list:
          #print(df['Class'][i])
          #majority.append(training_df_with_class['Rings'][i])
          #print(training_df_with_class['Class'][i])

        #class_decision = st.mode(majority)
        

        #print(class_decision[0])
        #decision.append(class_decision)


        thing1 = []
        thing2 = []

        denominator = []
        numerator = []

        for i in desicion_list:
          weight_holder = []
          for ii in desicion_list:
            if i != ii:
              w_i_part = 2.72**(-(i-ii)**2)
              weight_holder.append(w_i_part)
          w_i = sum(weight_holder)
          denominator.append(w_i)
          y_i_w_i = w_i * i
          numerator.append(y_i_w_i) 

          thing1.append(numerator/denominator)
          thing2.append(testing_df_with_class.iloc[i, -1])



      #counts = 0
      #for i in range(len(decision)):
        #thing1.append(decision[i][0][0])
        #thing2.append(testing_df_with_class.iloc[i, -1])
        #print(decision[i][0])
        #print(testing_df_with_class.iloc[i, -1])
        #testing_df_with_class['Rings'][i]
        #if decision[i][0] == testing_df_with_class.iloc[i, -1]:
          #counts += 1


        thing3.append(thing1)
        thing4.append(thing2)
    

      self.labels = thing4
      self.predictions = thing3

    print(self.labels)
    print(self.predictions)


      #print(counts)
      #print(counts/len(testing_df_with_class))

      #accuracies.append(counts/len(testing_df_with_class))
    #print(accuracies)

      #return [self.tuning, self.labels]
