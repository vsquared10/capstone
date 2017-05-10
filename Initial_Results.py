
# coding: utf-8

# In[3]:

import pandas as pd
import seaborn as sns
get_ipython().magic('pylab inline')


# In[4]:

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import matplotlib.pyplot as plt
import numpy as np


# In[5]:

data = ['developed_given_support_2015',
        'developing_received_support_2015',
        'food_security_index_2012',
        'usaid_food_relief_received']


# In[6]:

resources = [pd.read_csv('Data/{}'.format(i)) for i in data]


# In[7]:

give_sup = resources[0]
rec_sup = resources[1]
FSI = resources[2]
USAID = resources[3]


# ## Exploratory Data Analysis

# In[15]:

give_sup.head()


# In[16]:

rec_sup.head()


# In[20]:

rec_sup.describe()


# In[14]:

rec_sup.columns


# In[89]:

fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(15, 8.27)
Total_Estimate = sns.barplot(x='Country', y='Undernourishment', data=rec_sup, ax=ax)
for item in Total_Estimate.get_xticklabels():
    item.set_rotation(70)


# In[117]:

mu = rec_sup
mu = mu.loc[(mu!=0).any(axis=1)]
rec_sup_sample = pd.DataFrame(mu.sample(100))
µ = mu['Average Dietary Energy Supply Adequacy']
xbar = rec_sup_sample['Average Dietary Energy Supply Adequacy']
sns.distplot(µ, bins=25);


# ## Machine Learning Tests

# In[8]:

ignore = ['Unnamed: 0', 'Country', 'ISO3', 'Subcontinents', 'Groups',
       'Remote Country ISO3']


# In[9]:

x = rec_sup.drop(ignore, axis=1)
y = rec_sup['Food Safety Score']


# In[10]:

rf = RandomForestRegressor(n_estimators=100,
                           n_jobs=-1,
                           random_state=1)

gdbr = GradientBoostingRegressor(learning_rate=0.1,
                                 loss='ls',
                                 n_estimators=100,
                                 random_state=1)

abr = AdaBoostRegressor(DecisionTreeRegressor(),
                        learning_rate=0.1,
                        loss='linear',
                        n_estimators=100,
                        random_state=1)


# In[12]:

def get_scores(x,y,model):
    X_train, X_test, y_train, y_test = train_test_split(x,y, random_state = 1, test_size = 0.2)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    r2 = r2_score(y_test,pred)
    print('{0} Train CV | MSE: {1:.3f} | R2: {2:.3f}'.format(type(model).__name__,mse,r2))

get_scores(x,y,rf)
get_scores(x,y,gdbr)
get_scores(x,y,abr)


# In[164]:

gdbr_1 = GradientBoostingRegressor(learning_rate=1,
                                 loss='ls',
                                 n_estimators=100,
                                 random_state=1)
get_scores(x,y,gdbr_1)


# In[165]:

def stage_score_plot(model, train_x, train_y, test_x, test_y):
    '''
    INPUT:
     model: GradientBoostingRegressor or AdaBoostRegressor
     train_x: 2d numpy array
     train_y: 1d numpy array
     test_x: 2d numpy array
     test_y: 1d numpy array

    Create a plot of the number of iterations vs the MSE for the model for
    both the training set and test set.
    '''
    model.fit(train_x,train_y)
    train_scores = [mean_squared_error(train_y, train_predict) for train_predict in model.staged_predict(train_x)]
    test_scores = [mean_squared_error(test_y, test_predict) for test_predict in model.staged_predict(test_x)]
    plt.plot(train_scores, label = model.__class__.__name__ + " Train - learning rate " + str(model.learning_rate), ls = '--')
    plt.plot(test_scores,label = model.__class__.__name__ + " Test - learning rate " + str(model.learning_rate))
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title(model.__class__.__name__)


# In[166]:

train_x, test_x, train_y, test_y = train_test_split(x,y, random_state = 1, test_size = 0.2)
stage_score_plot(gdbr, train_x, train_y, test_x, test_y)
plt.legend()
plt.show()


# In[167]:

stage_score_plot(gdbr, train_x, train_y, test_x, test_y)
stage_score_plot(gdbr_1, train_x, train_y, test_x, test_y)
plt.legend()
plt.show()


# In[168]:

rf.fit(train_x, train_y)
p = rf.predict(test_x)
stage_score_plot(gdbr, train_x, train_y, test_x, test_y)
plt.axhline(mean_squared_error(test_y,p), label = 'Random Forest Regressor', ls='-.', c = 'y')
plt.legend()
plt.show()


# In[169]:

rf.fit(train_x, train_y)
p = rf.predict(test_x)
stage_score_plot(abr, train_x, train_y, test_x, test_y)
plt.axhline(mean_squared_error(test_y,p), label = 'Random Forest Regressor', ls='-.', c = 'y')
plt.legend()
plt.show() 


# In[170]:

random_forest_grid = {'max_depth': [3, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [1, 2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 20, 40],
                      'random_state': [1]}

rf_gridsearch = GridSearchCV(RandomForestRegressor(),
                             random_forest_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='neg_mean_squared_error')
rf_gridsearch.fit(train_x, train_y)

print("best parameters:", rf_gridsearch.best_params_)

best_rf_model = rf_gridsearch.best_estimator_


# In[171]:

best_rf_model.fit(train_x,train_y)
p1 = best_rf_model.predict(test_x)
mean_squared_error(test_y,p1)


# In[172]:

gradient_boosting_grid = {'learning_rate': [0.2,0.3],
                          'max_features': [0.5,0.51],
                          'max_depth': [4,5],
                          'min_samples_leaf': [3,4],
                          'n_estimators': [130,150],
                          'random_state': [1]}

gdbr_gridsearch = GridSearchCV(GradientBoostingRegressor(),
                             gradient_boosting_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='neg_mean_squared_error')
gdbr_gridsearch.fit(train_x, train_y)

print("best parameters:", gdbr_gridsearch.best_params_)

best_gdbr_model = gdbr_gridsearch.best_estimator_


# In[173]:

gdbr_gridsearch.best_params_


# In[174]:

best_gdbr_model.fit(train_x,train_y)
p2 = best_gdbr_model.predict(test_x)
mean_squared_error(test_y,p2)


# In[13]:

adaboosting_grid = {'learning_rate': [0.1],
                    'n_estimators': [20,40,50,100],
                    'loss': ['linear'],
                    'random_state': [1]}

abr_gridsearch = GridSearchCV(AdaBoostRegressor(DecisionTreeRegressor()),
                             adaboosting_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='neg_mean_squared_error')
abr_gridsearch.fit(train_x, train_y)

print("best parameters:", abr_gridsearch.best_params_)
best_abr_model = abr_gridsearch.best_estimator_


# In[176]:

best_abr_model.fit(train_x,train_y)
p3 = best_abr_model.predict(test_x)
mean_squared_error(test_y,p3)


# In[2]:

sample = test_x.iloc[0]


# In[1]:

sample


# In[188]:

sample_result = test_y.iloc[0]


# In[189]:

prediction_test = best_abr_model.predict(sample)
prediction_test


# In[190]:

sample_result


# ## More dataset exploration

# In[93]:

FSI.describe()


# In[140]:

USAID.head()


# In[138]:

total_received = pd.DataFrame(USAID.groupby(['Recipient Country', 'Fiscal Year'])["Obligations ($US)"].sum())


# In[139]:

total_received


# In[ ]:



