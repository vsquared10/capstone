
# coding: utf-8

# In[179]:

import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import numpy as np
get_ipython().magic('pylab inline')


# ## GFSI Data ETL

# In[410]:

data = ['food_security_index_201{}'.format(i) for i in range(2, 7)]


# In[411]:

resources = [pd.read_csv('../P8Data/{}'.format(i)) for i in data]


# In[412]:

resources[0]['year'] = 2012
resources[1]['year'] = 2013
resources[2]['year'] = 2014
resources[3]['year'] = 2015
resources[4]['year'] = 2016


# In[413]:

GFSI2012 = resources[0]
GFSI2013 = resources[1]
GFSI2014 = resources[2]
GFSI2015 = resources[3]
GFSI2016 = resources[4]


# In[427]:

dfs = [GFSI2012,
       GFSI2013,
#        GFSI2014,
       GFSI2015,
       GFSI2016]


# In[367]:

# eliminate nan in 2014
# pd.isnull(dfs[2]).values.any()
# pd.isnull(dfs[2]).any().nonzero()[0]
# dfs[2].drop(dfs[2].index[30], inplace=True)


# In[415]:

all_GFSI = pd.concat(dfs)


# In[416]:

pd.isnull(all_GFSI).any().nonzero()[0]


# In[417]:

for i in pd.isnull(all_GFSI).any().nonzero()[0]:
    all_GFSI.drop(all_GFSI.index[i], inplace=True)


# In[419]:

pd.isnull(all_GFSI).any().nonzero()[0]


# In[505]:

#pre-process data
y = GFSI2012['3.5.1) Agency to ensure the safety and health of food Qualitative assessment (0-1)']
X = GFSI2012.drop([y.name, 'Unnamed: 0', 'Metric Desc', 'year'], axis=1)
X = StandardScaler().fit_transform(X)


# In[506]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[223]:

lr = LogisticRegression()

rf = RandomForestClassifier(n_estimators=100,
                           n_jobs=-1,
                           random_state=1)

gdbr = GradientBoostingClassifier(learning_rate=0.1,
                                 loss='deviance',
                                 n_estimators=100,
                                 random_state=1)

abr = AdaBoostClassifier(DecisionTreeClassifier(),
                        learning_rate=0.1,
                        n_estimators=100,
                        random_state=1)

svc = SVC()

nn = KNeighborsClassifier()


# In[188]:

def get_scores(X_train, X_test, y_train, y_test, model):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    #prec = precision_score(y_test, pred)
    #rec = recall_score(y_test, pred)
    mse = mean_squared_error(y_test,pred)
    r2 = r2_score(y_test,pred)
    print('{0} Train CV:\n             Accuracy: {1:.3f}\n             MSE: {2:.3f}\n             R2: {3:.3f}\n'.format(type(model).__name__, accuracy, mse, r2))


# In[189]:

models = [lr, rf, gdbr, abr, svc, nn]


# In[190]:

for m in models:
    get_scores(X_train, X_test, y_train, y_test, m)


# In[256]:

lr_grid1 = {'penalty': ['l1', 'l2'],
           'tol': [0.0001, 0.001, 0.01, 0.1],
           'C': [.1,1,10,100],
           'fit_intercept': [True, False],
           'max_iter': [50, 100, 150],
           'multi_class': ['ovr'],
           'random_state': [1]}

lr_grid2 = {'penalty': ['l2'],
           'tol': [0.0001, 0.001, 0.01, 0.1],
           'C': [.1,1,10,100],
           'fit_intercept': [True, False],
           'solver': ['newton-cg', 'lbfgs', 'sag'],
           'max_iter': [50, 100, 150],
           'multi_class': ['ovr', 'multinomial'],
           'random_state': [1]}

rf_grid = {'n_estimators': [10, 20, 40],
           'max_depth': [2, 3, 10, None],
           'max_features': ['sqrt', 'log2', None],
           'min_samples_split': [0.5, 2, 4],
           'min_samples_leaf': [1, 2, 4],
           'bootstrap': [True, False],
           'random_state': [1]}

gdbr_grid = {'loss': ['deviance', 'exponential'],
             'learning_rate': [0.001, 0.1, 0.5],
             'n_estimators': [50, 100, 150],
             'subsample': [0.75, 1.0],
             'min_samples_split': [0.5, 2, 4],
             'min_samples_leaf': [1, 2, 4],
             'max_depth': [3, 10, None],
             'random_state': [1],
             'max_features': ['sqrt', 'log2', None]}

abr_grid = {'learning_rate': [0.001, 0.1, 0.5],
            'n_estimators': [20,40,50,100],
            'random_state': [1]}

svc_grid = {'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5],
            'gamma': ['auto', 0.1, 0.3, 1, 3, 10],
            'random_state': [1]}

nn_grid = {'n_neighbors': [5, 10, 20, 30],
           'weights': ['uniform', 'distance'],
           'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
           'leaf_size': [15, 30, 60],
           'p': [1, 2, 5]}


# In[507]:

def GS(train_x, test_x, train_y, test_y, model, grid):
    gridsearch = GridSearchCV(model,
                              grid,
                              n_jobs=-1,
                              verbose=True,
                              scoring='neg_mean_squared_error')
    gridsearch.fit(train_x, train_y)

    print("best parameters:", gridsearch.best_params_)

    best_model = gridsearch.best_estimator_
    get_scores(train_x, test_x, train_y, test_y, best_model)
    return best_model, gridsearch


# In[266]:

GS(X_train, X_test, y_train, y_test, lr, lr_grid1)


# ### LR_grid1 results:
# > LogisticRegression Train CV: Accuracy: 0.870 | MSE: 0.130 | R2: -0.150

# In[267]:

GS(X_train, X_test, y_train, y_test, lr, lr_grid2)


# ### LR_grid2 results:
# > LogisticRegression Train CV: Accuracy: 0.870 | MSE: 0.130 | R2: -0.150

# In[509]:

rf_estimator, rf_mod = GS(X_train, X_test, y_train, y_test, rf, rf_grid)


# ### RF_grid results:
# > RandomForestClassifier Train CV: Accuracy: 0.913 | MSE: 0.087 | R2: 0.233

# In[269]:

GS(X_train, X_test, y_train, y_test, gdbr, gdbr_grid)


# ### GB_grid results:
# > GradientBoostingClassifier Train CV: Accuracy: 0.870 | MSE: 0.130 | R2: -0.150

# In[270]:

GS(X_train, X_test, y_train, y_test, abr, abr_grid)


# ### AB_grid results:
# > AdaBoostClassifier Train CV: Accuracy: 0.870 | MSE: 0.130 | R2: -0.150

# In[271]:

GS(X_train, X_test, y_train, y_test, svc, svc_grid)


# ### SVC_grid results:
# > GradientBoostingClassifier Train CV: Accuracy: 0.870 | MSE: 0.130 | R2: -0.150

# In[510]:

kn_estimator, kn_mod = GS(X_train, X_test, y_train, y_test, nn, nn_grid)


# ### KNN_grid results:
# > KNeighborsClassifier Train CV: Accuracy: 0.913 | MSE: 0.087 | R2: 0.233

# # Final Testing
# 
# > Two best models were Random Forest and KNN. Test for better performance on total dataset

# In[511]:

#pre-process data
y_13 = GFSI2013['3.5.1) Agency to ensure the safety and health of food Qualitative assessment (0-1)']
X_13 = GFSI2013.drop([y.name, 'Unnamed: 0', 'Metric Desc', 'year'], axis=1)
X_13 = StandardScaler().fit_transform(X_13)


# In[512]:

X_train_13, X_test_13, y_train_13, y_test_13 = train_test_split(X_13, y_13, test_size=0.20, random_state=42)


# In[513]:

fin_rf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=2, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=40, n_jobs=-1, oob_score=False, random_state=1,
            verbose=0, warm_start=False)

fin_kn = KNeighborsClassifier(algorithm='auto', leaf_size=15, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=10, p=1,
           weights='uniform')

fin_rf_grid = {'n_estimators': [10, 20, 40],
           'max_depth': [2, 3, 10, None],
           'max_features': ['sqrt', 'log2', None],
           'min_samples_split': [0.5, 2, 4],
           'min_samples_leaf': [1, 2, 4],
           'bootstrap': [True, False],
           'random_state': [1]}

fin_kn_grid = {'n_neighbors': [5, 10, 20, 30],
           'weights': ['uniform', 'distance'],
           'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
           'leaf_size': [15, 30, 60],
           'p': [1, 2, 5]}


# In[516]:

finalmodels = [rf_estimator, kn_estimator]


# ## 2013 performance

# In[517]:

for m in finalmodels:
    get_scores(X_train_13, X_test_13, y_train_13, y_test_13, m)


# ## 2015 performance

# In[518]:

#pre-process data
y_15 = GFSI2015['3.5.1) Agency to ensure the safety and health of food Qualitative assessment (0-1)']
X_15 = GFSI2015.drop([y.name, 'Unnamed: 0', 'Metric Desc', 'year'], axis=1)
X_15 = StandardScaler().fit_transform(X_15)
X_train_15, X_test_15, y_train_15, y_test_15 = train_test_split(X_15, y_15, test_size=0.20, random_state=42)
for m in finalmodels:
    get_scores(X_train_15, X_test_15, y_train_15, y_test_15, m)


# ## 2016 performance

# In[519]:

#pre-process data
y_16 = GFSI2016['3.5.1) Agency to ensure the safety and health of food Qualitative assessment (0-1)']
X_16 = GFSI2016.drop([y.name, 'Unnamed: 0', 'Metric Desc', 'year'], axis=1)
X_16 = StandardScaler().fit_transform(X_16)
X_train_16, X_test_16, y_train_16, y_test_16 = train_test_split(X_16, y_16, test_size=0.20, random_state=42)
for m in finalmodels:
    get_scores(X_train_16, X_test_16, y_train_16, y_test_16, m)


# In[ ]:



