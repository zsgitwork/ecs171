### SVM MODEL ###

# Challenge: too many variables, SVM best for comparing 2 specific features

'''
refered to: https://scikit-learn.org/stable/modules/svm.html
and https://www.geeksforgeeks.org/support-vector-machine-algorithm/
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV  #library to split test/train data + find optimal params
from sklearn import svm  #library for SVM
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report  #library to calc MSE, accuracy, clasification report


# load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# drop 'id' column
df = df.drop(columns=['id'])

#display head of csv file (just to visualize)
print(df.head())

#fixing categories w/ non-number entries
categories = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df = pd.get_dummies(df, columns=categories)

# Fill missing vals of bmi with mean
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

#display head of csv file (just to visualize)
print(df.head())

# define X (data) and y (target)
X = df.drop(columns=['stroke'])  #input data = all features EXCEPT STROKE
y = df['stroke']  #target variable = stroke 

#split into training and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#intializing model
model = svm.SVC()

#fit model with training data
model.fit(X_train, y_train)

#make predictions with test data
y_pred = model.predict(X_test)

#calculating MSE
MSE = mean_squared_error(y_test, y_pred)
print("MSE: ", MSE)

#calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

#calculating classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

#note: displaying may be difficult bc so many features, will have to choose specific ones!