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

plt.rcParams['font.family'] = 'Times New Roman' #font for all plots 

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

#Create a 3D GRAPH
fig3D = plt.figure(figsize = [11, 14])
scatter3D = fig3D.add_subplot(111, projection='3d')
scatter = scatter3D.scatter(df['X'],df['Y'],df['Z'], c=df['Step'],cmap='tab20')
#axis labels
scatter3D.set_xlabel('X')
scatter3D.set_ylabel('Y')
scatter3D.set_zlabel('Z')
#title
scatter3D.set_title('Maintenance steps in 3D')
#legend
legend = scatter3D.legend(*scatter.legend_elements(), title="Step")
scatter3D.add_artist(legend)

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

#For evaluating model performance 
scoring_met = ['accuracy','precision','f1']

#Model 1: Random Forrest Classifier 
Model_RF = RandomForestClassifier()
Param_RF = {
    'n_estimators': [50, 100, 200],
}

GridSearch_RF = GridSearchCV(estimator=Model_RF, param_grid=Param_RF, scoring=scoring_met, refit='accuracy')
GridSearch_RF.fit(Train_X, Train_Y)
print("Optimized Parameters for Random Forest:\n", GridSearch_RF.best_params_)

# Model 2: Logistic regression
Model_LogReg = LogisticRegression()
Param_LogReg = {
    'penalty': ['l2', 'elasticnet'],
    'C': [0.01, 0.1, 1, 10],
}

GridSearch_LogReg = GridSearchCV(estimator=Model_LogReg, param_grid=Param_LogReg, scoring=scoring_met,refit='accuracy')
GridSearch_LogReg.fit(Train_X, Train_Y)
print("Optimized Parameters for Logistic Regression:\n", GridSearch_LogReg.best_params_)

# Model 3: Support Vector Machine (SVM)
Model_SVM = svm.SVC()
Param_SVM = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

GridSearch_SVM = GridSearchCV(estimator=Model_SVM,param_grid=Param_SVM,scoring=scoring_met, refit='accuracy')
GridSearch_SVM.fit(Train_X, Train_Y)
print("Optimized Parameters for SVM:\n", GridSearch_SVM.best_params_)

#Model 4: SVN - RandomizedSearchCV
Model_SVM_Rand = svm.SVC()
Param_SVM_Rand = {
    'C': np.linspace(0.01, 10, 20),
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

RandomSearch_SVM = RandomizedSearchCV(estimator=Model_SVM_Rand, param_distributions=Param_SVM_Rand, scoring=scoring_met, refit='accuracy')
RandomSearch_SVM.fit(Train_X, Train_Y)
print("Optimized Parameters for SVM model (RandomizedSearchCV):\n", RandomSearch_SVM.best_params_)

#Step 5: Model Performance Analysis
Model_RF = RandomForestClassifier(**GridSearch_RF.best_params_)
Model_LogReg = LogisticRegression(**GridSearch_LogReg.best_params_)
Model_SVM = svm.SVC(**GridSearch_SVM.best_params_)
Model_SVM_Rand = svm.SVC(**RandomSearch_SVM.best_params_)

# Models on the training data
Model_RF.fit(Train_X, Train_Y)
Model_LogReg.fit(Train_X, Train_Y)
Model_SVM.fit(Train_X, Train_Y)
Model_SVM_Rand.fit(Train_X, Train_Y)

# Predict in regards to the test set
Pred_RF = Model_RF.predict(Test_X)
Pred_LogReg = Model_LogReg.predict(Test_X)
Pred_SVM = Model_SVM.predict(Test_X)
Pred_SVM_Rand = Model_SVM_Rand.predict(Test_X)

# Classification Reports
Report_RF = classification_report(Test_Y, Pred_RF)
Report_LogReg = classification_report(Test_Y, Pred_LogReg)
Report_SVM = classification_report(Test_Y, Pred_SVM)
Report_SVM_Rand = classification_report(Test_Y, Pred_SVM_Rand)

print("Random Forest Classifier Metrics:\n", Report_RF)
print("Logistic Regression Metrics:\n", Report_LogReg)
print("Support Vector Machine (GridSearchCV) Metrics:\n", Report_SVM)
print("Support Vector Machine (RandomizedSearchCV) Metrics:\n", Report_SVM_Rand)

#SVM was chosen as it got all the predictions correct 
Model_Select = Model_SVM_Rand
Model_Pred = Pred_SVM_Rand
print("Support Vector Machine (RandomizedSearchCV) is the selected confusion matrix.")

#Confusion Matrix for SVM Rand Model 
Select_Conf = confusion_matrix(Test_Y, Model_Pred)
Label_Disp = ['1','2','3','4','5','6','7','8','9','10','11','12','13']
Disp_Select_Conf = ConfusionMatrixDisplay(confusion_matrix= Select_Conf, display_labels=Label_Disp)
ax = Disp_Select_Conf.plot(cmap='viridis')
plt.title("Confusion Matrix - SVM ((RandomizedSearchCV)",fontsize=16, fontname='Times New Roman')
plt.show()

#Step 6: Stacked Model Performance Analysis
#Logestic Regression and SVM were the chosen models for this  
Estimators_Stack = [
    ('LogReg_Base', LogisticRegression(**GridSearch_LogReg.best_params_))
]
Model_Stack = StackingClassifier(estimators=Estimators_Stack, final_estimator=svm.SVC(**GridSearch_SVM.best_params_))
Model_Stack.fit(Train_X, Train_Y)
Pred_Stack = Model_Stack.predict(Test_X)
# Generate and print classification report
Report_Stack = classification_report(Test_Y, Pred_Stack)
print("Stacked Model (Logistic Regression + SVM) Performance:\n", Report_Stack)
# Confusion matrix
ConfMatrix_Stack = confusion_matrix(Test_Y, Pred_Stack)
# Plot the confusion matrix
ConfDisplay_Stack = ConfusionMatrixDisplay(confusion_matrix=ConfMatrix_Stack, display_labels=Label_Disp)
ax = ConfDisplay_Stack.plot(cmap='Purples')
# Plot Formatting
plt.title("Confusion Matrix - Stacked Model (LogReg + SVM)", fontname='Times New Roman', fontsize=16)
plt.xlabel("Predicted Label", fontname='Times New Roman', fontsize=12)
plt.ylabel("True Label", fontname='Times New Roman', fontsize=12)
plt.xticks(fontname='Times New Roman', fontsize=10)
plt.yticks(fontname='Times New Roman', fontsize=10)
plt.show()

#Step 7: Model Evaluation 
#Model Name 
Model_Filename = 'SVM_Randomized_Model.joblib'
#Trained SVM rand gets svaed 
joblib.dump(Model_SVM_Rand, Model_Filename)
#Load Model 
Loaded_Model = joblib.load(Model_Filename)
#Data Given 
Given_Data = pd.DataFrame([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])
#Predict steps with the loaded model 
Predicted_Steps = Loaded_Model.predict(Given_Data)
#Print Predictions 
print("\nPredictions for the Data Given:")
print(Given_Data)
print("\nPredicted Maintenance Step:")
print(Predicted_Steps)
