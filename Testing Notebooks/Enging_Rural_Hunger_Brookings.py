
# coding: utf-8

# ### following csv's are as of Sep 15, 2015

# In[1]:

import pandas as pd


# In[3]:

developed = pd.read_csv('../../raw_dac.csv', index_col=None).fillna(value=0)


# In[4]:

developing = pd.read_csv('../../raw_developing-country.csv', index_col=None).fillna(value=0)


# In[5]:

developed.head()


# In[6]:

developing.head()


# In[8]:

developed.to_csv('../Data/developed_given_support_2015')


# In[9]:

developing.to_csv('../Data/developing_received_support_2015')

