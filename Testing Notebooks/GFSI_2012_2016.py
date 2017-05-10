
# coding: utf-8

# # GFSI Data cleaning and saving for years 2012-2016

# In[1]:

import pandas as pd
import glob


# In[19]:

path = '../../GFSI'
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
GFSI_dfs = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=None)
    GFSI_dfs.append(df)


# In[20]:

for i in range(len(GFSI_dfs)):
    GFSI_dfs[i][0] = GFSI_dfs[i][0] + ' ' + GFSI_dfs[i][1]
    GFSI_dfs[i].drop(1, axis=1, inplace=True)
    GFSI_dfs[i] = GFSI_dfs[i].T
    GFSI_dfs[i] = GFSI_dfs[i].dropna(axis=1, how='all')
    new_header = GFSI_dfs[i].iloc[0]
    GFSI_dfs[i] = GFSI_dfs[i][1:]
    GFSI_dfs[i].rename(columns = new_header, inplace=True)
    GFSI_dfs[i]['1.3) Gross domestic product per capita (US$ PPP) US$ at PPP / capita'] = GFSI_dfs[i]['1.3) Gross domestic product per capita (US$ PPP) US$ at PPP / capita'].apply(lambda x: x.replace(',', ''))
    GFSI_dfs[i]['2.1.1) Average food supply kcal/capita/day'] = GFSI_dfs[i]['2.1.1) Average food supply kcal/capita/day'].apply(lambda x: x.replace(',', ''))
    pd.to_numeric(GFSI_dfs[i]['1.1) Food consumption as a share of household expenditure % of total household expenditure']).mean()
    GFSI_dfs[i].ix[:, GFSI_dfs[i].columns != 'Metric Desc'] = GFSI_dfs[i].ix[:, GFSI_dfs[i].columns != 'Metric Desc'].apply(lambda x: x.replace('n/a', None))
    GFSI_dfs[i].ix[:, GFSI_dfs[i].columns != 'Metric Desc'] = GFSI_dfs[i].ix[:, GFSI_dfs[i].columns != 'Metric Desc'].apply(pd.to_numeric)
    GFSI_dfs[i].to_csv('../Data/food_security_index_201{}'.format(str(i + 2)))


# # Don't execute below code

# In[415]:

GFSI = pd.read_csv('../../GFSI_2012.csv', header=None)


# In[416]:

GFSI[0] = GFSI[0] + ' ' + GFSI[1]
GFSI.drop(1, axis=1, inplace=True)


# In[417]:

GFSI = GFSI.T


# In[418]:

GFSI = GFSI.dropna(axis=1, how='all')


# In[419]:

new_header = GFSI.iloc[0]
GFSI = GFSI[1:]
GFSI.rename(columns = new_header, inplace=True)
GFSI.head()


# In[420]:

GFSI['1.3) Gross domestic product per capita (US$ PPP) US$ at PPP / capita'] = GFSI['1.3) Gross domestic product per capita (US$ PPP) US$ at PPP / capita'].apply(lambda x: x.replace(',', ''))


# In[421]:

GFSI['2.1.1) Average food supply kcal/capita/day'] = GFSI['2.1.1) Average food supply kcal/capita/day'].apply(lambda x: x.replace(',', ''))


# In[423]:

GFSI.head()


# In[425]:

pd.to_numeric(GFSI['1.1) Food consumption as a share of household expenditure % of total household expenditure']).mean()


# In[426]:

GFSI.ix[:, GFSI.columns != 'Metric Desc'] = GFSI.ix[:, GFSI.columns != 'Metric Desc'].apply(lambda x: x.replace('n/a', None))


# In[427]:

GFSI.ix[:, GFSI.columns != 'Metric Desc'] = GFSI.ix[:, GFSI.columns != 'Metric Desc'].apply(pd.to_numeric)


# In[431]:

GFSI.tail()


# In[433]:

GFSI.to_csv('../Data/food_security_index_2012')

