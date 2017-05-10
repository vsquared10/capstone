
# coding: utf-8

# In[1]:

import requests


# In[16]:

url = 'http://api.aiddata.org/aid/project'

Filters = {
    'src' :'1,2,3,4,5,6,7,3249668',
    'from' : '0',
    'size' : '50'
}


# In[17]:

r = requests.get(url, params=Filters)


# In[18]:

r.json()


# In[19]:

data = r.json()


# In[20]:

data.keys()


# In[21]:

data['stats']


# In[22]:

data['project_count']


# In[ ]:



