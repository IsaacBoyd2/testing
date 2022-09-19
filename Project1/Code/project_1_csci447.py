#import Pandas and Numpy

import pandas as pd
import numpy as np
import math
import random

#--------------------------------------
#Bring in the data

df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/c55c311d66e9dd04da9a6ee8627facdcc11e9d2a/DataProject1/glass.csv?raw=true')#, on_bad_lines='skip')

#--------------------------------------

#This should Probably be converted into a different dataframe instead of variables in order to make it more visual.

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
#print(testing_df)

#random_list = random.choices(possible_values, k=len(df))

#print(random_list)
#print(len(random_list))

#sum = training_size + testing_size
#print(sum)


#print(F_values)

#bins2 = range(len(bins))

#print(bins)
#column_names = list(df.columns)
#print(column_names)
#y = 0
#bin_number = 0

#print(len(testing_df.columns))

results = []

for lines in range(len(testing_df)):
  row = testing_df.iloc[lines]
  C_x = []
  for i in bins:   
    F_a_c_list = []                                                 #For the categories get a small df
    category_df = training_df[training_df['Glass Type'] == i]         #get all of the first category
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
    
  print(C_x)
  results.append(bins[C_x.index(max(C_x))])

print(results)
#print(testing_df_with_labels[0:20])

    #index = C_x.index(max(C_x))

    
      #print(denominator)

    

#print(C_x)
      

  #now we are going to take each training peice and categorize it



  #training_df = 

  #globals()[f"F{i}"] = pd.DataFrame(index= range(len(category_df)), columns = range(len(column_names)-1)) #make an identical dataset with 

  #print(category_df[1])
  #print(category_df)
  #for lines in range(len(category_df)):                          #in each df for every row
    #print(lines)
    #row = category_df.iloc[lines]
    #print(row)
    #print(lines)
    #for count,attributes in enumerate(row):                        #then for every attribute in this row
      #x = 0
      #y = 0
      #for k in category_df.iloc[:, count][0:len(category_df)]:        #loop through the category_df's rows and get a y
        #if k == attributes:
          #y = y + 1
      #print(count)
      #F_values.at[lines, bin_number] = y
      #print(bin_number)
      #F_values[lines, bin_number] = y                        # Put that y value in the df.
      #print(F_values)  
      


  #bin_number += 1

#print(F_values)
  #print('kadjhfkajshdfkahsdf')
  #print(bin_number)
  
        #print('hello')

      #print(attributes)
      #y = y + 1



      #try:
      #  x = category_df.value_counts(attributes)
      #  print('hello')
      #except Exception:
      #  x = 0
      #x = category_df.value_counts(attributes)
      
      #print(x)

#print(y)

    #print(attributes)


#Get an attribute value see how many are the same within a class plus one and then divide by the number in the class plus the number of atributes.

  


#print(bins)

#for i in range(1, max(df['Glass Type'])+1):                        #Reset the global variables on each run
#  globals()[f"number_of_{i}s"] = 0

#surveys_df[surveys_df.year == 2002]

#data_list = []
#max_list = []

#for i in range(1, max(df['Glass Type'])+1):                        #For all possible categories sum all of the categories and then divide
#  for j in range(len(df)):                                       #by the length of the dataframe. Store these values in a variable.
#    if df['Glass Type'][j] == i:                                 # --  Q(C = ci) = #{x ∈ ci}/N
#      globals()[f"number_of_{i}s"] += 1
#  max_list.append(globals()[f"number_of_{i}s"])
#  globals()[f"Q{i}"] = globals()[f"number_of_{i}s"]/len(df)
#  data_list.append(globals()[f"Q{i}"])



#for i in range(1, max(df['Glass Type'])+1):
  

#data_list = np.array(data_list)

#Q_values = pd.DataFrame(data= [data_list], columns = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7'])

# (Aj = ak, C = ci) = (#{(xAj = ak) ∧ (x ∈ ci)} + 1) /(Nci + d)

#Get an attribute value see how many are the same within a class plus one and then divide by the number in the class plus the number of atributes.

#F_Values = pd.DataFrame(index= range(max(max_list)), columns = range(len(data_list)))



#F_Values
  
#print(Q1)
#print(number_of_2s)
