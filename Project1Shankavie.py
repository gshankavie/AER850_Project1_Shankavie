#AER850 - Project 1
#Intro to Machine Learning  
#Owner: Shankavie Gnaneswaran
#Submission Date: Oct 11, 2025

#Set up the imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  

#Step 1: Data Visualization
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

#Step 2: Plot and Explain the Data 
#Create a LINE GRAPH
#Create a HISTOGRAM
df.plot(linestyle = '-')
plt.title('Linegraph of the DataFrames 4 Columns',fontsize=16, fontname='Times New Roman')
plt.xlabel('DataFrame Row Number',fontname='Times New Roman')
plt.ylabel('Values',fontname='Times New Roman')
plt.show()

df.hist( color=['m'],edgecolor = 'black')
plt.xlabel('Values',fontname='Times New Roman')
plt.ylabel('Number of Occurences',fontname='Times New Roman')
plt.suptitle('Histograms of DataFrames 4 Columns',fontsize=16, fontname='Times New Roman')
plt.show()

#Step 3: Correlation Analysis 
#Goal of analyzing the correlations between the various coloumns
#Plot heatmap to visualize the correlations, use seaborn  
plt.figure() #new figure for plotting 
analysis_corr = data.corr() #computes pearson correlation coefficient in DataFrame
sns.heatmap(np.abs(analysis_corr)) #plotting heatmap, takes absolute values of correlations
#selects the Step column and the X column, finds the pearson correlation coefficient
#takes the absolute value 
corr_ana_X = np.abs(df.iloc[:,3].corr(df.iloc[:,0])) 
print("")



#Step 4: Classification Model Development and Engineering 

