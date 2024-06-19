#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import zipfile
import os


# In[2]:


# Define the path to the zip file and the extraction directory
zip_file_path = r"C:\Users\chara\Downloads\creditcard.csv.zip"
extracted_dir_path = r"C:\Users\chara\Downloads"

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir_path)

# Load the dataset
csv_file_path = os.path.join(extracted_dir_path, "creditcard.csv")
data = pd.read_csv(csv_file_path)

# Display the first few rows of the dataset
print(data.head())


# In[3]:


# Check for missing values
print(data.isnull().sum())

# Define features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[4]:


# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# In[5]:


# Create and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test)

# Evaluate the model
print("Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))
print(confusion_matrix(y_test, y_pred_log_reg))


# In[ ]:


# Create and train the Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_clf.predict(X_test)

# Evaluate the model
print("Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))


# Random Forest Classifier:
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00     56901
#            1       1.00      1.00      1.00     56887
# 
#     accuracy                           1.00    113788
#    macro avg       1.00      1.00      1.00    113788
# weighted avg       1.00      1.00      1.00    113788
# 
# [[56892     9]
#  [    1 56886]]
# 

# In[ ]:




