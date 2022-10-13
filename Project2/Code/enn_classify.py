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
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats as st
import requests
import os
import random as random
import sys

#----Python Classes import----

classInputArray = [['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/preprocessing.py?raw=true','preprocessing.py'], 
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/model.py?raw=true','model.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/model_regression.py?raw=true','model_regression.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/loss.py?raw=true', 'loss.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/k_means_classify.py?raw=true', 'k_means_classify.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/k_means_regress.py?raw=true', 'k_means_regress.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/enn_classify.py?raw=true', 'ennc.py']]


for i in range(len(classInputArray)):
  with open(classInputArray[i][1], 'w') as f:
    r = requests.get(classInputArray[i][0])
    f.write(r.text)

import preprocessing as pp
import model as classifier
import model_regression as regressor
import loss as lss
import k_means_classify as kmeans
import k_means_regress as kmeansr
import ennc as enncc

#################################

k_nn = 3

preProcess = pp.Preprocessing()
preProcess.process()
preProcess.fold()
data = [preProcess.tuning, preProcess.folds]
    
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


  #print(training_df)
  #print(testing_df)

  
  old_val_acc = 0
  val_acc = 1

  clicks = 1

  tracker = [0,0]

  #The test is when the validation accuracy or the accuract when testing against the tuning dataset falls below the previous break out.

  while (tracker[clicks] > tracker[clicks-1] or clicks==1):

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

    df_matrix = pd.DataFrame(np.nan, index=range(len(tuning_df)), columns = range(len(training_df)))

    for count1 in range(len(tuning_df)): #go throught the tuning df
      base = tuning_df.iloc[count1]
      for count2 in range(len(training_df)): #compare against training df
        dist1 = []
        for count3 in range(len(tuning_df.columns)):     
            dist1.append(((base[count3]) - (training_df.iloc[count2][count3]))**2)   #Find the distance   #1 error right here
             

      
        summation = sum(dist1)   #this seesm fine
        distance = np.sqrt(summation)   #this is fine for now as long as you are not getting a wierd error

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

    #print(len(decision))
    #print(len(tuning_df_with_class))
    #print(counts)

    val_acc = counts/len(tuning_df_with_class)   #calculate validation accuracy

    tracker.append(val_acc)
    #print(tracker)
    #print("Validation_accuracy",val_acc)

    #the above code was to give us a comparison as to when to stop

    ############################### Now we edit #######################################

    clicks = clicks + 1 #increment 1
    

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
      

      comparison_array = np.array(df_matrix.loc[count1])
      k = k_nn
      index = np.argpartition(comparison_array, k)
      reduced_idx = index[:k]

      reduced_list = reduced_idx.tolist()

      #print(reduced_list)

      
      

      majority =[]
      #print(counting)
      #print(training_df_with_class)
      for i in reduced_list:
        
        #print(i-counting)

        majority.append(training_df_with_class.iloc[i- counting, -1])   #removed counting. Maybe now that drop is true we do not need to reset the counting or something?


      class_decision = st.mode(majority)
      pd.set_option('display.max_rows', None)
      pd.set_option('display.max_columns', None)
      pd.set_option('display.width', None)
      pd.set_option('display.max_colwidth', -1)

      
      

      if class_decision[0] != training_df_with_class.iloc[count1 - counting, -1]:
        print(training_df_with_class)
        #print(training_df[0:10])
        training_df = training_df.drop(count1 - counting)
        training_df_with_class = training_df_with_class.drop(count1- counting)
        counting = counting + 1

        #print(len(training_df_with_class))

        #print(training_df_with_class)

        training_df_with_class = training_df_with_class.reset_index()
        training_df = training_df.reset_index()

        training_df_with_class = training_df_with_class.drop(columns=['index'])
        training_df = training_df.drop(columns=['index'])

        print(training_df_with_class)

        sys.exit()

        #training_df_with_class = training_df_with_class.drop(columns=['level_0'])
        #raining_df = training_df.drop(columns=['level_0'])

      

  


  df_matrix = pd.DataFrame(np.nan, index=range(len(training_df)), columns = range(len(training_df)))

  decision = []

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

  print(thing3)
  thing4.append(thing2)

print(thing4)
print(thing3)



self.labels = thing4
self.predictions = thing3

print(self.labels)
print(self.predictions)
