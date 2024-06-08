import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from IPython.display import display
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
# load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
#display first five subjects
display(df.head(5))

df = df.drop(columns=['id'])

categories = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df = pd.get_dummies(df, columns=categories)


numerical_columns = ['age', 'avg_glucose_level', 'bmi']

print("\nColumns with empty values: ", df.columns[df.isna().any()]) # bmi
# # Fill bmi with mean
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
print(df.shape)

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



# df['stroke_label'] = df['stroke'].map({0: 'No Stroke', 1: 'Stroke'})

# # Group the data by gender and stroke status, and count the occurrences
# grouped_data = df.groupby(['gender', 'stroke_label']).size().unstack()

# # Plotting the stacked bar chart
# grouped_data.plot(kind='bar', stacked=True)
# plt.title('Stroke Cases by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.xticks(rotation=0)  # Rotate x-axis labels if needed
# plt.legend(title='Stroke', labels=['No Stroke', 'Stroke'])
# plt.show()

#Random Forest Model-Divya 

#normalize data
num = ["age", "avg_glucose_level", "bmi"]
scaler = StandardScaler()
df[num] = scaler.fit_transform(df[num])

#create data frame with featues and target 
X = df.drop('stroke', axis=1)
y = df['stroke']

#split into training and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(df)
print(X)

#apply smote 
smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier()

# param_grid = {
#     'n_estimators': [300, 500, 1000],
#     'max_depth': [5, 10, None],
# }
# grid_search = GridSearchCV(rf, param_grid)
# grid_search.fit(x_smote, y_smote)

# best_rdf = grid_search.best_estimator_
# y_pred = best_rdf.predict(X_test)

# print("RF Optimal Accuracy with Grid Search: ", grid_search.best_score_)
# print("RF Test Accuracy: ", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# # Visualization of Random Forest Model through ROC Curve
# # Calculate the probabilities for the ROC curve
# y_prob = best_rdf.predict_proba(X_test)[:, 1]

# # Calculate the false positive rate and true positive rate for the ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# # Calculate the area under the ROC curve
# auc = roc_auc_score(y_test, y_prob)

# # Plot the ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(auc))
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# # plt.show()

# from joblib import dump

# dump(best_rdf, "./utils/model.joblib")
# dump(scaler, "./utils/scaler.joblib")