#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[7]:




# In[11]:


#before giving as input and output just do the encoding for the three columns
from sklearn.preprocessing import LabelEncoder
sex_encoder=LabelEncoder()
df["sex_encoded"] = sex_encoder.fit_transform(df["Sex"])


# In[12]:


embarked_encoder=LabelEncoder()
df["embark_encoded"] = sex_encoder.fit_transform(df["Embarked"])


# In[13]:


deck_encoder=LabelEncoder()
df["deck_encoded"] = deck_encoder.fit_transform(df["Deck"])


# In[14]:




# In[15]:



# In[17]:


#train and create linear regression model
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)


# In[18]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)


# In[19]:


import joblib

joblib.dump(model, 'titanic_model.pkl')


# In[22]:


joblib.dump(sex_encoder, 'sex_encoder.pkl')
joblib.dump(embarked_encoder, 'embark_encoder.pkl')
joblib.dump(deck_encoder, 'deck_encoder.pkl')


# In[ ]:




