#import Pandas and Numpy

import pandas as pd
import numpy as np

#--------------------------------------
#Bring in the data

df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/c55c311d66e9dd04da9a6ee8627facdcc11e9d2a/DataProject1/glass.csv?raw=true')#, on_bad_lines='skip')

#--------------------------------------

#This should Probably be converted into a different dataframe instead of variables in order to make it more visual.

bins = df['Glass Type'].unique() #May need to rename each one to be the same accross datasets. Instead of "Glass Type" maybe do "categorical".
                                  #Also we may need a binning alg if data is continuous

F_Values = pd.DataFrame(index= range(len(df)), columns = range(len(bins)))

print(bins)
y = 0
for i in bins:
  category_df = df[df['Glass Type'] == i]
  #print(category_df[1])
  #print(category_df)
  for lines in range(len(category_df)):
    #print(lines)
    row = category_df.iloc[lines]
    #print(row)
    #print(lines)
    for count,attributes in enumerate(row):
      x = 0
      y = 0
      for k in category_df.iloc[:, count][0:len(category_df)]:
        if k == attributes:
          y = y + 1
      print(y)
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

print(y)

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
