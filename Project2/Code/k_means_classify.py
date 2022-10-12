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

  def run(self, data, k_nn, k_cluster, value):

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


      #weights should be means not choices really, but if our means do not change neither should the clusters, so this is potneitally valid


    ####################### Go until the weights are the same ##################################

      while (Initial_weights != new_weights):

        #should start as empty. Gains new wieghts when algorithm is run.

        Initial_weights = new_weights.copy()

        if clicks == 0:

          #starts as empty
          weights_matrix = pd.DataFrame(np.nan, index=range(k_cluster), columns = range(len(training_df.columns)))

          #get some random values to use as indexes
          randomList = random.sample(range(len(training_df)), k_cluster)

          #Set the weight matrixes up with the training data
          for counters,i in enumerate(randomList):
            weights_matrix.iloc[counters] = training_df.iloc[i]

        clicks = clicks + 1

        #print(weights_matrix)   #seems good up to here populates with 35 different centroids



        #Create an empty dataframe called df matrix for our new centroids
        df_matrix = pd.DataFrame(np.nan, index=range(len(weights_matrix)), columns = range(len(training_df)))

        #create a holder to hold all of our points.
        Centroid_holder = []
        for i in range(k_cluster):
            Centroid_holder.append([])

        #For everthing in our matrix  

        for count1 in range(len(weights_matrix)):
          #Grab the rows 1 by 1
          base = weights_matrix.iloc[count1]
          #Iterate through the training data.
          for count2 in range(len(training_df)):
            dist1 = []
            #Go through every column soo that we hit all the atributes
            for count3 in range(len(weights_matrix.columns)):
                #Find the distance between every centroid and traing point.
                dist1.append(((base[count3]) - (training_df.iloc[count2][count3]))**2)

            #Get the summation of the distance of all atributes.        
            summation = sum(dist1)

            #take the square root
            distance = summation**0.5



            df_matrix.iloc[count1, count2] = distance 

        #print(df_matrix) #Also seems to be good.
        #print(clicks)    #seems to be good

        for i in df_matrix.columns:
          new_weights[i] = df_matrix.iloc[:,i].idxmin()  #check if updates
          Centroid_holder[int(df_matrix.iloc[:,i].idxmin())].append(i) #this will append the indexs of each closest training datum to each centroid section which is what we are trying to do

        for i in range(len(Centroid_holder)):
          #Create a blank dataframe
          df_holder = pd.DataFrame(np.nan, index=range(len(Centroid_holder[i])), columns = range(len(training_df.columns)))
          for the_count,ii in enumerate(Centroid_holder[i]):
            df_holder.iloc[the_count] =  training_df.iloc[ii]   #check to make sure everything makes it into here

          df2_holder = pd.DataFrame(np.nan, index=range(1), columns = range(len(training_df.columns)))

          for iii in range(len(df_holder.columns)):
            df2_holder[iii] = df_holder[iii].mean()    #find the mean of each atrribute

          #p = df2_holder.to_numpy()    #I don't see a point where we would genearte a nan

          #if np.isnan(p).all() == False:

          weights_matrix.iloc[i] = df2_holder #new centroids


    ########################################       Labeling          #######################################################

      weights_matrix_labels = weights_matrix.copy()    #create a copy of the wieghts dataframe

      weights_matrix_labels['Class'] = 1     #populate its "class" with all 1s

      #create another empty dataframe
      df_matrix = pd.DataFrame(np.nan, index=range(len(weights_matrix)), columns = range(len(training_df)))

      decision = []

      for count1 in range(len(weights_matrix)):
        base = weights_matrix.iloc[count1]
        for count2 in range(len(training_df)):
          dist1 = []
          for count3 in range(len(weights_matrix.columns)):
              dist1.append(((base[count3]) - (training_df.iloc[count2][count3]))**2)   #find distance between each centroid and the training points

          summation = sum(dist1)  #sum all attributes

          distance = summation**0.5   #take the sqrt

          df_matrix.loc[count1, count2] = distance 

        comparison_array = np.array(df_matrix.loc[count1])  #For every centoid get a comparison matrix

        index = np.argpartition(comparison_array, k_nn)     #Find the k closest values to it.
        reduced_idx = index[:k_nn]                   

        reduced_list = reduced_idx.tolist()                #Turn into list


        majority =[]
        for i in reduced_list:
          majority.append(training_df_with_class.iloc[i, -1])

        class_decision = st.mode(majority)

        weights_matrix_labels['Class'][count1] = class_decision[0]  #compare this with actuals

      print(weights_matrix_labels)   #these seem fairly reasonable
      #print(training_df)

      #sys.exit()

      df_matrix = pd.DataFrame(np.nan, index=range(len(testing_df)), columns = range(len(weights_matrix)))

      decision = []

      for count1 in range(len(testing_df)):
        base = testing_df.iloc[count1]      #for every testing value
        # if base > 10**5:
              #= 0
        for count2 in range(len(weights_matrix)):
          dist1 = []
          for count3 in range(len(testing_df.columns)):
              dist1.append(((base[count3]) - (weights_matrix.iloc[count2][count3]))**2)   #compared to the centroids    #one line of code wrong.


          summation = sum(dist1)

          distance = summation**0.5

          df_matrix.loc[count1, count2] = distance 


        comparison_array = np.array(df_matrix.loc[count1])
        #k = k_nn

        index = np.argpartition(comparison_array, k_nn)

        reduced_idx = index[:k_nn]

        reduced_list = reduced_idx.tolist()

        print(reduced_list)

        majority =[]
        for i in reduced_list:
          majority.append(weights_matrix_labels.iloc[i, -1])

        #print(majority)

        class_decision = st.mode(majority)

        decision.append(class_decision)

      counts = 0

      thing1 = []
      thing2 = []

      for i in range(len(decision)):
        thing1.append(decision[i][0][0].item())

        thing2.append(testing_df_with_class.iloc[i, -1])

      thing3.append(thing1)

      thing4.append(thing2)
