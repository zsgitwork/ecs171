# ECS171
 This project explores the potential of using machine learning models to predict the likelihood of strokes. The unpredictability of strokes poses significant challenges in identifying individuals at risk, and current methods of stroke risk assessment lack the necessary accuracy and speed. Our project addresses these issues by developing a machine learning model optimized for stroke risk assessment. The dataset used in this study, referred to as the Stroke Prediction dataset [1], contains health-related information on 5,110 patients. Excluding patient ID numbers, this dataset includes 11 health-related features, which we utilize to analyze attribute correlations and train a machine learning model capable of predicting an individual’s likelihood of getting strokes. The final model will be integrated into a user-friendly website where individuals can input their health data and receive personalized stroke risk assessments, making the assessment process more accessible and efficient for the public. By leveraging the power of machine learning, we aim to provide a more reliable and faster method for assessing stroke risk.

Our EDA code is in:
- EDA (original): model.py in the main branch.
- EDA (expanded): eda.py in the LogisticRegression-Shruthi branch.
  
To view our tested models, please refer to their appropriate branches:
- Random Forest Model (optimal): located in the model.py file in the RandomForest-Divya branch. 
- Logistic Regression Model: located in the logisticreg.py file in the LogisticRegression-Shruthi branch.
- Support Vector Machine Model: located in the SVM_model.py file in the MadelynNguyen_SVM_model branch.

We also created a branch for testing the SMOTE preprocessing algorithm:
- SMOTE testing: located in the model.py file in the SMOTE branch. 

For the frontend implementation (website), please navigate to: 
- frontend.py in the frontend branch
