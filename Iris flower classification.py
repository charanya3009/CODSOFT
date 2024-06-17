#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sns


# In[16]:


df = sns.load_dataset('iris')
df.head()


# In[17]:


df['species'],categories =pd.factorize(df['species'])
df.head()


# In[18]:


df.describe


# In[19]:


df.isna().sum()


# In[23]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(df.petal_length, df.petal_width, df.species)
ax.set_xlabel('Petal length Cm')
ax.set_ylabel('Petal width Cm')
ax.set_zlabel('Species')
plt.title('3D Scatter Plot Example')
plt.show()


# In[24]:


sns.scatterplot(data=df, x="sepal_length", y="sepal_width",hue="species")


# In[25]:


sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")


# In[26]:


k_rng = range(1,10)
sse=[]

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal_length','petal_width']])
    sse.append(km.inertia_)


# In[27]:


sse


# In[28]:


plt.xlabel('k_rng')
plt.ylabel("sum of squared errors")
plt.plot(k_rng,sse)


# In[31]:


km= KMeans(n_clusters=3,random_state=0,)
y_predicted = km.fit_predict(df[['petal_length','petal_width']])
y_predicted


# In[32]:


df['cluster']=y_predicted
df.head(150)


# In[33]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(df.species,df.cluster)
cm


# In[ ]:




