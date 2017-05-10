
# coding: utf-8

# In[1]:

reset -fs


# In[2]:

import pandas as pd


# In[5]:

USloans = pd.DataFrame.from_csv('../../US_Overseas_Loans.csv')


# In[6]:

USloans.head()


# In[7]:

USloans['Program Name'].unique()


# In[8]:

Food_aid = USloans[USloans['Program Name'].str.contains('Food')]


# In[16]:

Food_aid.head()


# In[11]:

Food_aid.to_csv('../Data/usaid_food_relief_received')

