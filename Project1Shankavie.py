#AER850 - Project 1
#Intro to Machine Learning  
#Owner: Shankavie Gnaneswaran
#Submission Date: Oct 10, 2025

#Set up the imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

#Load CSV file into DataFrame
df = pd.read_csv('Project 1 data.csv') 

#Preview the given data 
print("Data Types and Coloumn Names:\n")
print(df.info()) #data types and coloumn names 
print("First 5 Rows of the Data:\n")
print(df.head()) #first 5 rows of the data
print("Number of Rows and Coloumns:\n")
print(df.shape) #number of rows and coloumns 
print("Statistics for the Coloumns:\n")
print(df.describe()) #statistics for the coloumns

