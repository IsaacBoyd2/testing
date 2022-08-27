import pandas as pd
import numpy as np
import math
import random

#--------------------------------------
#Bring in the data

df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/c55c311d66e9dd04da9a6ee8627facdcc11e9d2a/DataProject1/glass.csv?raw=true')

#--------------------------------------

#Preprocessing

bins = df['Glass Type'].unique() #May need to rename each one to be the same accross datasets. Instead of "Glass Type" maybe do "categorical".
                                  #Also we may need a binning alg if data is continuous
print(bins)

training_size = math.ceil(len(df)*4/5)
#testing_size = math.floor(len(df)*1/5)

random_list = random.sample(range(len(df)), len(df))


training_list = random_list[0:training_size]
testing_list = random_list[training_size:len(df)]

training_df =  df.iloc[training_list]
testing_df_with_labels = df.iloc[testing_list]
testing_df = testing_df_with_labels.iloc[: , 1:-1]


#------------------------------------

results = []

for lines in range(len(testing_df)):
  row = testing_df.iloc[lines]
  C_x = []
  for i in bins:   
    F_a_c_list = []                                                 #For the categories get a small df
    category_df = training_df[training_df['Glass Type'] == i]         #get all of the first category
    #print(len(category_df))
    for count,j in enumerate(row):
      y = 0
      for k in category_df.iloc[:, count][0:len(category_df)]:        #loop through the category_df's rows and get a y
        if k == j:
          y = y + 1
        
      numerator = y + 1
      denominator = len(category_df)+len(testing_df.columns)-1
      
      F_a_c = numerator/denominator
      F_a_c_list.append(F_a_c)

    C_x.append(np.prod(F_a_c_list)*(len(category_df)/len(df)))
    
  #print(C_x)
  results.append(bins[C_x.index(max(C_x))])


correct = 0
for count,i in enumerate(results):
  if testing_df_with_labels['Glass Type'].iloc[count] == i:
    correct += 1

accuracy = correct/len(results)
print(accuracy)

#print(results)
#print(testing_df_with_labels['Glass Type'])
