#AER850 - Project 1
#Intro to Machine Learning  
#Owner: Shankavie Gnaneswaran
#Submission Date: Oct 11, 2025

#Set up the imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

#Load CSV file into DataFrame
df = pd.read_csv('Project 1 data.csv') 

#Preview the given data 
print("Data Types and Columns Names:\n")
print(df.info()) #data types and columns names 
print("First 5 Rows of the Data:\n")
print(df.head()) #first 5 rows of the data
print("Number of Rows and Columns:\n")
print(df.shape) #number of rows and columns 
print("Statistics for the Columns:\n")
print(df.describe()) #statistics for the columns

#Plot and Explain the Data 
#Create a LINE GRAPH
#Create a HISTOGRAM
#creates a new blank figure, 10 in wide and 8 in tall  
plt.figure(figsize=(10,8))
df.plot()
df.hist()






#create 1 histogram per variable X, Y, Z and see how they are distributed for each step 
# for col in ['X','Y','Z']:
#     sns.histplot(df, x=col, hue='Step', bins=15,kde=True,element='step')
#     plt.title('{col} Distribution per Step')
#     plt.show()
