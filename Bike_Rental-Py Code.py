#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading libraries
import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randrange, uniform
import KNN


# In[2]:


#Changing the directory
os.chdir("/Users/rishi/Desktop/All/edWisor/Project1")


# In[3]:


#Importing the dataset
bike_datamain =  pd.read_csv("day.csv", index_col = 0)


# ## Missing Value Analysis

# In[4]:


missingval = pd.DataFrame(bike_datamain.isnull().sum())
missingval = (missingval/len(bike_datamain))*100
missingval.reset_index()
#Checkec for NULL values (if any present in the dataset)
missingval = missingval.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
missingval = missingval.sort_values('Missing_percentage', ascending = False)
missingval.to_csv("Missing_perc.csv", index = False) #saving the missing values to the dataset
missingval
##No any missing values found in our dataset


# ## Outlier Analysis

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(bike_datamain['hum'])
#Negetive outlier found in 'hum' variable


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(bike_datamain['windspeed'])
#Positive outliers found in 'windspeed' variable


# In[11]:


cnames = ["dteday","yr","season","mnth","workingday","weekday","weathersit","temp","atemp","hum","windspeed"]
pnames = ["temp","hum","windspeed"]


# In[8]:


#Detecting & Deleting the Outliers
for i in pnames :
    print (i)
    q75,q25 = np.percentile(bike_datamain.loc[:,i],[75,25])
    iqr = q75-q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print (min)
    print (max)
    
    bike_datamain = bike_datamain.drop(bike_datamain[bike_datamain.loc[:,i] < min].index)
    bike_datamain = bike_datamain.drop(bike_datamain[bike_datamain.loc[:,i] > max].index)   


# ## Feature Engineering

# In[12]:


#Converting variables to required data types 
bike_datamain['dteday'] = pd.to_datetime(bike_datamain['dteday'],yearfirst=True)
bike_datamain['season'] = bike_datamain['season'].astype('category')
bike_datamain['yr'] = bike_datamain['yr'].astype('category')
bike_datamain['mnth'] = bike_datamain['mnth'].astype('category')
bike_datamain['holiday'] = bike_datamain['holiday'].astype('category')
bike_datamain['weekday'] = bike_datamain['weekday'].astype('category')
bike_datamain['workingday'] = bike_datamain['workingday'].astype('category')
bike_datamain['weathersit'] = bike_datamain['weathersit'].astype('category')
bike_datamain['temp'] = bike_datamain['temp'].astype('float')
bike_datamain['atemp'] = bike_datamain['atemp'].astype('float')
bike_datamain['hum'] = bike_datamain['hum'].astype('float')
bike_datamain['windspeed'] = bike_datamain['windspeed'].astype('float')
bike_datamain['casual'] = bike_datamain['casual'].astype('float')
bike_datamain['registered'] = bike_datamain['registered'].astype('float')
bike_datamain['cnt'] = bike_datamain['cnt'].astype('float')


# ## Feature Selection

# In[13]:


#to check for multicollinearity
#Correlation Plot
df_corr = bike_datamain.loc[:,cnames]
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generating correlation matrix
corr = df_corr.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[14]:


#Chi Square Test of Independence to check multicollinearity between categorical variables
#filtering categorical variables
cat_names = ["season","yr","mnth","holiday","weekday","workingday","weathersit"]


# In[16]:


from scipy.stats import chi2_contingency
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(bike_datamain['cnt'], bike_datamain[i]))
    print(dof)


# In[17]:


#Dropping the unwanted variables like atemp is highly correlated with temp, and others!
bike_datamain = bike_datamain.drop(['atemp','holiday','workingday','casual','registered'], axis =1)


# ## Model Development

# In[21]:


#Applying Random Forest Model
#Importing required libraries
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


# In[22]:


#Dividing the dataset into train and test data
X = bike_datamain.values[:,1:9]
Y = bike_datamain.values[:,9]

X_train,y_train,X_test,y_test = train_test_split( X, Y, test_size = 0.2)


# In[23]:


from sklearn.ensemble import RandomForestRegressor
RF_model = RandomForestRegressor(n_estimators = 1000, random_state = 1337)
# Running the model on training data
RF_model.fit(X_train, X_test);

#Predicting on Test case
predictions = RF_model.predict(y_train)
# Calculating the absolute errors
errors = abs(predictions - y_test)


# In[24]:


# Calculate MAPE (mean absolute percentage error)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


#Random Forect Model implemented successfully with appreciable accuracy.

