
# coding: utf-8

# # Unicef Data
# - children assumed to by under 5 years of age
# 
# ### Malnutrition
# - Severe Wasting: Percentage of children aged 0–59 months who are below minus three standard deviations from median weight-for-height of the WHO Child Growth Standards.
# - Wasting – Moderate and severe: Percentage of children aged 0–59 months who are below minus two standard deviations from median weight-for-height of the WHO Child Growth Standards.
# - Overweight – Moderate and severe: Percentage of children aged 0-59 months who are above two standard deviations from median weight-for-height of the WHO Child Growth Standards. 
# - Stunting – Moderate and severe: Percentage of children aged 0–59 months who are below minus two standard deviations from median height-for-age of the WHO Child Growth Standards.
# - Underweight – Moderate and severe: Percentage of children aged 0–59 months who are below minus two standard deviations from median weight-for-age of the World Health Organization (WHO) Child Growth Standards.
# 
# ### Stunting
# - percentages in terms of total child population

# In[1]:

import pandas as pd


# In[33]:

get_ipython().magic('pinfo pd.read_csv')


# In[19]:

malnutrition = pd.read_csv('../UNICEF_child_malnutrition_dec_2016.csv', header=1)


# In[20]:

malnutrition.head()


# In[29]:

stunting_area = pd.read_csv('../UNICEF_stunting_prevalence_by_area_residence_09_15.csv').dropna()


# In[30]:

stunting_area.head()


# In[31]:

stunting_wealth = pd.read_csv('../UNICEF_stunting_prevalence_children_by_wealth_quintile_09_15.csv').dropna()


# In[32]:

stunting_wealth.head()

