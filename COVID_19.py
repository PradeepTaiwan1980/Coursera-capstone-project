#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pingouin as pg


# In[18]:


import pingouin as pg
import os
import pandas as pd
from pandas.plotting import scatter_matrix

#Directory where data-file has been saved:
flpth = r"C:\LocalData\COVID19_Forecasting"
#File name of data-file:
flnm = "corrdatax01.xlsx"
#Read data into pandas dataframe:
data = pd.read_excel(os.path.join(flpth,flnm))
data.head()


# In[19]:


#Plot histogram of data:
data.hist(bins=50, figsize=(18,14))
#Plot scatter matrix of data:
scatter_matrix(data, figsize=(18,14))
#Find correlation matrix of data:
corr_matrix = data.corr()
#Print out correlations of Output w.r.t. others:
print(corr_matrix[data.columns[0]].sort_values(ascending=False))


# In[20]:


import pingouin as pg
pg.pairwise_corr(data).sort_values(by=['p-unc'])[['X', 'Y', 'n', 'r', 'p-unc']].head()


# In[21]:


#Pairwise correlation
import pingouin as pg
from pingouin import pairwise_corr, read_dataset
data[['Output (X0)', 'Tourism Score (X2)', 'Economic Score (X1)','SARS Score (X3)', 'Human Freedom Score']].pcorr()


# In[22]:


# Partial Correlations Matrix of variables 
data.pcorr().sort_values(data.columns[0], ascending=False)


# In[23]:


# Partial Correlations Matrix of variables (not in order, for easy comparison with JMP results)
data.pcorr()


# In[24]:


#Correlation Matrix 
data.corr().round(2)


# In[64]:


# Correlation matrix. 
# Warm colors (red) indicate a positive correlation, cool colors (blue) indicate a negative correlation

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
corrs = data.corr()
mask = np.zeros_like(corrs)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corrs, cmap='Spectral_r', mask=mask, square=True, vmin=-.4, vmax=.4)
plt.title('Correlation Matrix-1')


# In[26]:


# Display Correlation Matrix with "p-values with 5 decimal places"
data.rcorr(stars=False, decimals=6)


# In[27]:


import os
import pandas as pd
from pandas.plotting import scatter_matrix

#Directory where data-file has been saved:
flpth1 = r"C:\LocalData\COVID19_Forecasting"
#File name of data-file:
flnm1 = "COVID_Temp_Humidity.xlsx"
#Read data into pandas dataframe:
data1 = pd.read_excel(os.path.join(flpth1,flnm1))
data1.head()


# In[28]:


# Partial Correlations Matrix of variables (not in order, for easy comparison with JMP results)
data1.pcorr()


# In[65]:


#Correlation Matrix 
data1.corr().round(5)


# In[66]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
corrs = data1.corr()
mask = np.zeros_like(corrs)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corrs, cmap='Spectral_r', mask=mask, square=True, vmin=-.4, vmax=.4)
plt.title('Correlation Matrix-2')


# In[31]:


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-09-2020.csv')


# In[34]:


latest_data.head()


# In[35]:


confirmed_df.head()


# In[36]:


cols = confirmed_df.keys()


# In[82]:


confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]


# In[83]:


days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)


# In[84]:


days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]


# In[85]:


start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# In[86]:


X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.05, shuffle=False)


# In[87]:


# svm_confirmed = svm_search.best_estimator_
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)


# In[88]:


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forcast, svm_pred, linestyle='dashed', color='red')
plt.title('# of Coronavirus Cases Over Time', size=20)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


# In[89]:


# Future predictions using SVM 
svm_df = pd.DataFrame({'Date': future_forcast_dates[-5:], 'SVM Predicted # of Confirmed Cases Worldwide': np.round(svm_pred[-5:])})
svm_df


# In[2]:


import pingouin as pg
import os
import pandas as pd
from pandas.plotting import scatter_matrix

#Directory where data-file has been saved:
flpth2 = r"C:\LocalData\COVID19_Forecasting\Comparison" 
#File name of data-file:
flnm2 = "COVID_Data.xlsx"
#Read data into pandas dataframe:
df = pd.read_excel(os.path.join(flpth2,flnm2))
df.head()


# In[3]:


# Partial Correlations Matrix of variables (not in order, for easy comparison with JMP results)
df.pcorr()


# In[4]:


#Correlation Matrix 
df.corr().round(5)


# In[66]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
corrs = df.corr()
mask = np.zeros_like(corrs)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(25,10))
sns.heatmap(corrs, cmap='Spectral_r', mask=mask, square=True, vmin=-.4, vmax=.4)
plt.title('Correlation Matrix-3')


# In[5]:


# Display Correlation Matrix with "p-values with 5 decimal places"
df.rcorr(stars=False, decimals=6)


# In[1]:


import pingouin as pg
import os
import pandas as pd
from pandas.plotting import scatter_matrix

#Directory where data-file has been saved:
flpth3 = r"C:\LocalData\COVID19_Forecasting\Comparison" 
#File name of data-file:
flnm3 = "COVID_Data1.xlsx"
#Read data into pandas dataframe:
newdf = pd.read_excel(os.path.join(flpth3,flnm3))
newdf.head()


# In[2]:


newdf = newdf.drop(['Temperature', 'Humdity (%RH)', 'Tests per Million', 'Economic Score', 'SARS Score', 'EPI Score'] , axis='columns')
newdf.head()


# In[6]:


y = newdf['COVID-19 Cases'].values
x = newdf.drop('COVID-19 Cases', axis=1).values
x


# In[10]:


y = newdf.iloc[:,0:1] # first column of data frame (first_name)
y


# In[13]:


y = newdf['COVID-19 Cases'].values  # Keep Target y as 1D array of values for SKlearn Algorithms 
y


# In[12]:


x = newdf.iloc[:, 1:8] # second to eigth column of data frame (first_name), 
# keep independent variables as data frame, and normalize it after train_split   
x


# In[14]:


#Split Train and Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=10)
X_train


# In[19]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
X_train= preprocessing.StandardScaler().fit(X_train).transform(X_train)
X_train


# In[20]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
X_test= preprocessing.StandardScaler().fit(X_test).transform(X_test)
X_test


# In[15]:


y_train


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier


# In[27]:


newdf.head()


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(newdf.drop('COVID-19 Cases',axis=1), newdf['COVID-19 Cases'], test_size=0.30, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:


#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


# Bagged Decision Trees for Classification
kfold = model_selection.KFold(n_splits=10, random_state=10)
model_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=10)
results_1 = model_selection.cross_val_score(model_1, x, y, cv=kfold)
print(results_1.mean())


# In[32]:


# Random Forest Classification
kfold_rf = model_selection.KFold(n_splits=10)
model_rf = RandomForestClassifier(n_estimators=100, max_features=5)
results_rf = model_selection.cross_val_score(model_rf, x, y, cv=kfold_rf)
print(results_rf.mean())


# In[ ]:


#Adabooster
from sklearn.ensemble import AdaBoostClassifier
kfold_ada = model_selection.KFold(n_splits=10, random_state=10)
model_ada = AdaBoostClassifier(n_estimators=30, random_state=10)
results_ada = model_selection.cross_val_score(model_ada, x, y, cv=kfold_ada)
print(results_ada.mean())


# In[ ]:


#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
kfold_sgb = model_selection.KFold(n_splits=10, random_state=10)
model_sgb = GradientBoostingClassifier(n_estimators=100, random_state=10)
results_sgb = model_selection.cross_val_score(model_sgb, x, y, cv=kfold_sgb)
print(results_sgb.mean())


# In[ ]:


#Voting Ensemble
kfold_vc = model_selection.KFold(n_splits=10, random_state=10)
 
# Lines 2 to 8
estimators = []
mod_lr = LogisticRegression()
estimators.append(('logistic', mod_lr))
mod_dt = DecisionTreeClassifier()
estimators.append(('cart', mod_dt))
mod_sv = SVC()
estimators.append(('svm', mod_sv))
 
# Lines 9 to 11
ensemble = VotingClassifier(estimators)
results_vc = model_selection.cross_val_score(ensemble, x, y, cv=kfold_vc)
print(results_vc.mean())


# In[ ]:


#Neural Network
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))


# In[ ]:


print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))


# In[ ]:


Xgboost R-Sq = 0.65, Correlation = 0.9
Random Forest R-Sq = 0.64, Correlation = 0.9
Gradient Boosted Trees R-Sq = 0.61, Correlation = 0.86
Ridge Regression R-Sq = 0.56, Correlation = 0.88

