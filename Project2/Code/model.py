#-----------------------imports-------------------------
import pandas as pd
import numpy as np
from scipy import stats as st

class model:

  def __init__(self):
    df = pd.Dataframe
    predictions = []
    labels = []

  def run(self, data):

    #Hyper parameters
    k_nn = 3

    
    #first we want to talk data[1] and from it we can seperate it into its folds
    
    folds = data[1]

    #now we are going to run each fold and hold out one.

    for iterations in range(10):
      all_folds = range(10)
      all_folds.pop(iterations)

      testing_data = folds[iterations]

      training_data = []
      #for allof in all_folds:
      #  for peices in 



    #all_folds = range(10)

    

    #training = folds[1]





    #df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/glass.csv?raw=true')

    '''df_matrix = pd.DataFrame(np.nan, index=range(len(df)), columns = range(len(df)))


    df = df.drop(0)
    df = df.reset_index()
    df2 = df.iloc[: , 2:-1]

    decision = []

    for count1 in range(len(df2)):
        base = df2.iloc[count1]
        for count2 in range(len(df2)):
          dist1 = []
          for count3 in range(len(df2.columns)):
            dist1.append((base[count3] - df2.iloc[count2][count3])**2)
          
          summation = sum(dist1)
          distance = np.sqrt(summation)

          df_matrix.loc[count1, count2] = distance 

        comparison_array = np.array(df_matrix.loc[count1])
        k = k_nn+1
        index = np.argpartition(comparison_array, k)
        reduced_idx = index[:k]

        reduced_list = reduced_idx.tolist()

        reduced_list.remove(count1)

        majority =[]
        for i in reduced_list:
          #print(df['Class'][i])
          majority.append(df['Class'][i])

        class_decision = st.mode(majority)
        

        #print(class_decision[0])
        decision.append(class_decision)

    counts = 0
    for i in range(len(decision)):
      #print(decision[i][0])
      #print(df['Class'][i])
      if decision[i][0] == df['Class'][i]:
        counts += 1

    print(counts)
    print(counts/len(df))


    return [self.tuning, self.labels]'''
  
