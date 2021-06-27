#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Train=pd.read_csv("C:\\Users\\dell\\Desktop\\train.csv")


# In[52]:


#dataset information
Train.info()


# survival:    Survival 
# PassengerId: Unique Id of a passenger. 
# pclass:    Ticket class     
# sex:    Sex     
# Age:    Age in years     
# sibsp:    # of siblings / spouses aboard the Titanic     
# parch:    # of parents / children aboard the Titanic     
# ticket:    Ticket number     
# fare:    Passenger fare     
# cabin:    Cabin number     
# embarked:    Port of Embarkation

# In[53]:


#Check for null values
Train.isnull().sum()


# In[54]:


# Remove unwanted features
Train.drop('Cabin', axis=1, inplace=True)


# In[55]:


Train.info()


# In[56]:


# Filling null values using fillna method
Train.fillna(Train.mean(), inplace= True)


# In[57]:


#Again check for null values 
Train.isnull().sum()


# In[58]:



Train['Embarked'].value_counts(dropna=False)
Train['Embarked'].head()


# In[59]:


Train['Embarked']=Train['Embarked'].fillna(method='ffill')
Train['Embarked'].value_counts(dropna=False)


# In[60]:


Train.head()


# In[61]:


# correlation method
Train.corr()


# In[62]:


# Create new column adding the values of 2 columns SibSp and Parch
Train['Family_size'] = Train['SibSp'] + Train['Parch']


# In[63]:


#Drop the SibSp and Parch column 
Train.drop(['SibSp','Parch'], axis=1, inplace=True)


# In[64]:


Train.corr()


# # Data Analysis

# In[29]:


# Did gender paly role in survival
sns.countplot(Train['Sex'],hue= Train['Survived'])


# In[30]:


# Did class play role in survival
sns.countplot(Train['Pclass'],hue= Train['Survived'])


# In[42]:


# Did family bond play role in survival
sns.countplot(Train['Family_size'],hue= Train['Survived'])


# In[32]:


# total passengers survived
Train[Train['Survived']==1]['PassengerId'].nunique()


# In[33]:


# total male passengers in the ship
Train[Train['Sex']=='male']['PassengerId'].nunique()


# In[34]:


# total male passengers survived
Train[(Train['Sex']=='male')&(Train['Survived']==1)]['PassengerId'].nunique()


# In[35]:


# total female passengers in the ship
Train[Train['Sex']=='female']['PassengerId'].nunique()


# In[36]:


# total female passengers survived
Train[(Train['Sex']=='female')&(Train['Survived']==1)]['PassengerId'].nunique()


# In[44]:


Train.groupby(['Sex'])['Survived'].mean() # survival rate based on genger


# In[40]:


Train['Fare'].max() # maximum fare rate 


# In[41]:


Train['Fare'].min() # minmum fare rate


# In[46]:


Train.groupby(['Embarked'])['Survived'].mean() # survival rate based on Port of Embarkation


# In[47]:


Train.groupby(['Pclass'])['Survived'].mean() # survival rate based on class


# # CONCLUSION:

# 1.Gender play major role in survival as female passengers have higher survival rate than men. 
# 2.People in first class also have higher survival rate compare to other class. 
# 3.Passenger travelling with family also have higher survival rate than a passenger travelling alone.
# 4.Passenger who borded the ship at "C" have high survival range

# In[ ]:





# In[ ]:




