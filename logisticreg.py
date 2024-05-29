import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Display the first five rows of the dataset
print(data.head())

# Pre-process the data
# Drop the id column
data = data.drop('id', axis=1)
# Handle missing values
# List of numerical columns
numerical_columns = ['age', 'avg_glucose_level', 'bmi']

# Handle missing values for numerical columns
for col in numerical_columns:
    # Fill missing values with the mean of each column
    data[col].fillna(data[col].mean(), inplace=True)

# Encode categorical variables
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Split the data into features (X) and target variable (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split the data into training and testing sets 75:25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualization of Logistic Regression Model through ROC Curve
# Calculate the probabilities for the ROC curve
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Calculate the false positive rate and true positive rate for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_test, y_pred_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()