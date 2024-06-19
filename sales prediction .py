#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[16]:


# Load the dataset
data = pd.read_csv(r"C:\Users\chara\Downloads\advertising.csv")


# Display the first few rows of the dataset
print(data.head())


# In[17]:


# Check for missing values
print(data.isnull().sum())

# Define features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


# In[18]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# In[20]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[21]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[22]:


# Plot the actual vs predicted sales
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()


# In[ ]:




