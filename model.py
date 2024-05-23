### SVM MODEL ###

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm  # library for SVM

# load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#display head of csv file (just to visualize)
print(df.head())

# define X (data) and y (target)  #FIXME: not working rn bc data is in df
X = df.data
y = df.target

#intializing model
model = svm.SVC()



'''
#display first five subjects
display(df.head(5))

#correlation matrix for numeric features
numeric_columns = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()
print("Correlation matrix for numeric features")
print(correlation_matrix)

#heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

#print mean, median, std, etc. for numeric features
print("Describe all numeric features:")
print(df.describe())

#print mean, mediam, std, etc. based off if subject had a stroke or not
stroke_positive = df[df['stroke'] == 1]
stroke_negative = df[df['stroke'] == 0]
print("\nHad stroke:")
print(stroke_positive.describe())
print("\nDid not have stroke:")
print(stroke_negative.describe())

#plot age
avg_age_stroke = df[df['stroke'] == 1]['age'].mean()
avg_age_no_stroke = df[df['stroke'] == 0]['age'].mean()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
plt.bar(['Stroke', 'No Stroke'], [avg_age_stroke, avg_age_no_stroke], color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Average Age of Individuals with and Without Stroke')
plt.xlabel('Stroke')
plt.ylabel('Average Age')
plt.show()

avg_bmi_stroke = df[df['stroke'] == 1]['bmi'].mean()
avg_bmi_no_stroke = df[df['stroke'] == 0]['bmi'].mean()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
plt.bar(['Stroke', 'No Stroke'], [avg_bmi_stroke, avg_bmi_no_stroke], color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Average BMI of Individuals with and Without Stroke')
plt.xlabel('Stroke')
plt.ylabel('Average BMI')
plt.show()

#plotting average glucose level

avg_gl_stroke = df[df['stroke'] == 1]['avg_glucose_level'].mean()
avg_gl_no_stroke = df[df['stroke'] == 0]['avg_glucose_level'].mean()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
plt.bar(['Stroke', 'No Stroke'], [avg_gl_stroke, avg_gl_no_stroke], color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Average Glucose Level of Individuals with and Without Stroke')
plt.xlabel('Stroke')
plt.ylabel('Average Glucose Level')
plt.show()


df['stroke_label'] = df['stroke'].map({0: 'No Stroke', 1: 'Stroke'})

# Group the data by gender and stroke status, and count the occurrences
grouped_data = df.groupby(['gender', 'stroke_label']).size().unstack()

# Plotting the stacked bar chart
grouped_data.plot(kind='bar', stacked=True)
plt.title('Stroke Cases by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.legend(title='Stroke', labels=['No Stroke', 'Stroke'])
plt.show()
'''
