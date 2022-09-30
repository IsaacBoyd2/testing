#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: K-nearest neighbor
##Completed: 9-29-2022
##References: NA

#-----------------------imports-------------------------
import pandas as pd
import requests

#----Python Classes import----

r = requests.get('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Code/preprocessing.py?raw=true')

with open('preprocessing.py', 'w') as f:
    f.write(r.text)

import preprocessing as pp

#------------------------Main--------------------------

def main():
    preProcess = pp.Preprocessing()
    preProcess.process()

main()