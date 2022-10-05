#-----------------------imports-------------------------
import pandas as pd
import numpy as np
import requests
from scipy import stats as st

#Hyper parameters
k_nn = 3


df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/glass.csv?raw=true')

df_matrix = pd.DataFrame(np.nan, index=range(len(df)), columns = range(len(df)))


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


#Next we get the smallest values in the rows     Done
#Then we find the associated slice for the mins.  Done
#Then we find the associated classes
#Then we do a majority vote and classify.
#Then we find loss.


#For each slice. Find the distance between all other points. 

#Slice 1: dif 1,dif2, ... , dif99
#Slide 2:
#Slide 3:
#...
#Slice 99:
