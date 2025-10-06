#AER850 - Project 1
#Intro to Machine Learning  
#Owner: Shankavie Gnaneswaran
#Submission Date: Oct 11, 2025

#Set up the imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import joblib

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
df.plot(linestyle = '-')
plt.title('Linegraph of the DataFrames 4 Columns',fontsize=16, fontname='Times New Roman')
plt.xlabel('DataFrame Row Number',fontname='Times New Roman')
plt.ylabel('Values',fontname='Times New Roman')
plt.show()
#Create a HISTOGRAM
df.hist( color=['m'],edgecolor = 'black')
plt.xlabel('Values',fontname='Times New Roman')
plt.ylabel('Number of Occurences',fontname='Times New Roman')
plt.suptitle('Histograms of DataFrames 4 Columns',fontsize=16, fontname='Times New Roman')
plt.show()

#Step 3: Correlation Analysis 
#Goal of analyzing the correlations between the various coloumns
#Plot heatmap to visualize the correlations, use seaborn  
plt.figure() #new figure for plotting 
plt.title('Heatmap to Visualize X, Y, Z, and Step Correlations',fontsize=16, fontname='Times New Roman')
analysis_corr = df.corr() #computes pearson correlation coefficient in DataFrame
sns.heatmap(np.abs(analysis_corr)) #plotting heatmap, takes absolute values of correlations
#selects the Step column and the X column, finds the pearson correlation coefficient
#takes the absolute value 
corr_ana_X = np.abs(df.iloc[:,3].corr(df.iloc[:,0])) 
print("Step vs X Correlation:\n",corr_ana_X)
#selects the Step column and the Z column, finds the pearson correlation coefficient
#takes the absolute value
corr_ana_Z = np.abs(df.iloc[:,3].corr(df.iloc[:,2])) 
print("Step vs Z Correlation:\n",corr_ana_Z) 
#selects the Step column and the Y column, finds the pearson correlation coefficient
#takes the absolute value
corr_ana_Y = np.abs(df.iloc[:,3].corr(df.iloc[:,1])) 
print("Step vs Y Correlation:\n",corr_ana_Y) 

#1 represents data that is exactly the same
#comparing it to the step we do want high correlation, so it is easier to fit a model to it 
#within X Y Z we need it to be unique  
#X has the highest correlation to Step vs Z which has the lowest correlation with Step 
#Model should be trained with X data against Step 

#Step 4: Classification Model Development and Engineering
#Dataset is being split into testing and training sets  
my_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in my_splitter.split(df, df['Step']):
    Data_strat_train = df.iloc[train_index].reset_index(drop=True)
    Data_strat_test = df.iloc[test_index].reset_index(drop=True)
#Define the X and Y variables 
Train_Y = Data_strat_train['Step']
Train_X = Data_strat_train.drop('Step', axis=1)
Test_Y = Data_strat_test['Step']
Test_X = Data_strat_test.drop('Step', axis=1)

