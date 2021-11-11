#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


# In[78]:


#Load data
data_all = pd.read_csv('Olympics_Athletics_Results_cleaned.csv')
data_all.drop(columns=['Unnamed: 0'], inplace=True)


# In[150]:


data_2016


# In[231]:


#Split 2016 apart, only gold medalist, and create training and testing groups
data = data_all[data_all['Year'] != 2016]
data = data[data['Position'] == 1]

data_2016 = data_all[data_all['Year'] == 2016]
data = data[data['Position'] == 1]

#Separate by gender
data_f = data[data['Gender'] == 'F']
data_m = data[data['Gender'] == 'M']

#Create X and y
X_f = data_f.loc[(data_f['Event'] == '100 metres'), ['Year']]
y_f = data_f.loc[(data_f['Event'] == '100 metres'), ['Time']]

X_m = data_m.loc[(data_m['Event'] == '100 metres'), ['Year']]
y_m = data_m.loc[(data_m['Event'] == '100 metres'), ['Time']]


# In[181]:


data_2016.loc[((data_2016['Event'] == '100 metres') 
             & (data_2016['Gender'] == 'F')
             & (data_2016['Position'] == 1)), 'Time'].values[0]


# In[245]:


#Linear Regressor
from sklearn.linear_model import LinearRegression

#Fit the model
lr_f = LinearRegression()
lr_m = LinearRegression()

lr_f.fit(X_f, y_f)
lr_m.fit(X_m, y_m)

#Graph female times
plt.title('Female 100m Times in the Olympics (Linear)')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.scatter(X_f, y_f, color='blue')
plt.xticks(np.arange(1920, 2024, 10))
plt.scatter(2016, lr_f.predict([[2016]]), color='red')
plt.plot(X_f, lr_f.predict(X_f), color='green')
plt.show()

#Print female prediction and stats
prediction = lr_f.predict([[2016]])[0][0]
actual = data_2016.loc[((data_2016['Event'] == '100 metres') 
                      & (data_2016['Gender'] == 'F')
                      & (data_2016['Position'] == 1)), 'Time'].values[0]

print('Correlation Coefficient: {:0.2f}'.format(lr_f.score(X_f, y_f)))
print('2016 Prediction: {:0.2f} seconds'.format(prediction))
print('2016 Actual Time: {0} seconds'.format(actual))
print('Error: {:0.3f}%'.format(1 - abs(prediction / actual)))

#Graph male times
plt.title('Male 100m Times in the Olympics (Linear)')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.scatter(X_m, y_m, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, lr_m.predict([[2016]]), color='red')
plt.plot(X_m, lr_m.predict(X_m), color='green')
plt.show()

#Print male prediction and stats
prediction = lr_m.predict([[2016]])[0][0]
actual = data_2016.loc[((data_2016['Event'] == '100 metres') 
                      & (data_2016['Gender'] == 'M')
                      & (data_2016['Position'] == 1)), 'Time'].values[0]

print('Correlation Coefficient: {:0.2f}'.format(lr_m.score(X_m, y_m)))
print('2016 Prediction: {:0.2f} seconds'.format(prediction))
print('2016 Actual Time: {0} seconds'.format(actual))
print('Error: {:0.3f}%'.format(1 - abs(prediction / actual)))


# In[232]:


#Polynomial Regression
#Linear Regressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Transform and fit the model
poly_reg_f = LinearRegression()
poly_reg_m = LinearRegression()

poly_reg = PolynomialFeatures(degree = 4)
X_f_poly = poly_reg.fit_transform(X_f)
X_m_poly = poly_reg.fit_transform(X_m)

poly_reg_f.fit(X_f_poly, y_f)
poly_reg_m.fit(X_m_poly, y_m)

#Graph female times
plt.title('Female 100m Times in the Olympics (Polynomial)')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.scatter(X_f, y_f, color='blue')
plt.xticks(np.arange(1920, 2024, 10))
plt.scatter(2016, poly_reg_f.predict([[2016]]), color='red')
plt.plot(X_f, poly_reg_f.predict(X_f_poly), color='green')
plt.show()

#Print prediction and stats
prediction = poly_reg_f.predict([[2016]])[0][0]
actual = data_2016.loc[((data_2016['Event'] == '100 metres') 
                      & (data_2016['Gender'] == 'F')
                      & (data_2016['Position'] == 1)), 'Time'].values[0]

print('Correlation Coefficient: {:0.2f}'.format(poly_reg_f.score(X_f, y_f)))
print('2016 Prediction: {:0.2f} seconds'.format(prediction))
print('2016 Actual Time: {0} seconds'.format(actual))
print('Error: {:0.3f}%'.format(1 - abs(prediction / actual)))

#Graph male times
plt.title('Male 100m Times in the Olympics (Polynomial)')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.scatter(X_m, y_m, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, poly_reg_m.predict([[2016]]), color='red')
plt.plot(X_m, poly_reg_m.predict(X_m), color='green')
plt.show()

#Print prediction and stats
prediction = poly_reg_m.predict([[2016]])[0][0]
actual = data_2016.loc[((data_2016['Event'] == '100 metres') 
                      & (data_2016['Gender'] == 'M')
                      & (data_2016['Position'] == 1)), 'Time'].values[0]

print('Correlation Coefficient: {:0.2f}'.format(poly_reg_m.score(X_m, y_m)))
print('2016 Prediction: {:0.2f} seconds'.format(prediction))
print('2016 Actual Time: {0} seconds'.format(actual))
print('Error: {:0.3f}%'.format(1 - abs(prediction / actual)))


# In[246]:


#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

#Fit the model
dtr_f = DecisionTreeRegressor(random_state=42)
dtr_m = DecisionTreeRegressor(random_state=42)

dtr_f.fit(X_f, y_f)
dtr_m.fit(X_m, y_m)

#Graph female times
plt.title('Female 100m Times in the Olympics (Decision Tree)')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.scatter(X_f, y_f, color='blue')
plt.xticks(np.arange(1920, 2024, 10))
plt.scatter(2016, dtr_f.predict([[2016]]), color='red')
plt.plot(X_f, dtr_f.predict(X_f), color='green')
plt.show()

#Print female prediction and stats
prediction = dtr_f.predict([[2016]])[0]
print(prediction)
actual = data_2016.loc[((data_2016['Event'] == '100 metres') 
                      & (data_2016['Gender'] == 'F')
                      & (data_2016['Position'] == 1)), 'Time'].values[0]

print('Correlation Coefficient: {:0.2f}'.format(dtr_f.score(X_f, y_f)))
print('2016 Prediction: {:0.2f} seconds'.format(prediction))
print('2016 Actual Time: {0} seconds'.format(actual))
print('Error: {:0.3f}%'.format(1 - abs(prediction / actual)))

#Graph male times
plt.title('Male 100m Times in the Olympics (Decision Tree)')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.scatter(X_m, y_m, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, dtr_m.predict([[2016]]), color='red')
plt.plot(X_m, dtr_m.predict(X_m), color='green')
plt.show()

#Print male prediction and stats
prediction = dtr_m.predict([[2016]])[0]
actual = data_2016.loc[((data_2016['Event'] == '100 metres') 
                      & (data_2016['Gender'] == 'M')
                      & (data_2016['Position'] == 1)), 'Time'].values[0]

print('Correlation Coefficient: {:0.2f}'.format(dtr_m.score(X_m, y_m)))
print('2016 Prediction: {:0.2f} seconds'.format(prediction))
print('2016 Actual Time: {0} seconds'.format(actual))
print('Error: {:0.3f}%'.format(1 - abs(prediction / actual)))


# In[244]:


#Support Vector Regression
from sklearn.svm import SVR

svr_f = SVR()
svr_f.fit(X_f, y_f)

svr_m = SVR(kernel='rbf')
svr_m.fit(X_m, y_m)

#Graph female times
plt.title('Female 100m Times in the Olympics (SVR)')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.scatter(X_f, y_f, color='blue')
plt.xticks(np.arange(1920, 2024, 10))
plt.scatter(2016, svr_f.predict([[2016]]), color='red')
plt.plot(X_f, svr_f.predict(X_f), color='green')
plt.show()

#Print female prediction and stats
prediction = svr_f.predict([[2016]])[0]
actual = data_2016.loc[((data_2016['Event'] == '100 metres') 
                      & (data_2016['Gender'] == 'F')
                      & (data_2016['Position'] == 1)), 'Time'].values[0]

print('Correlation Coefficient: {:0.2f}'.format(svr_f.score(X_f, y_f)))
print('2016 Prediction: {:0.2f} seconds'.format(prediction))
print('2016 Actual Time: {0} seconds'.format(actual))
print('Error: {:0.3f}%'.format(1 - abs(prediction / actual)))

#Graph male times
plt.title('Male 100m Times in the Olympics (SVR)')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.scatter(X_m, y_m, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, svr_m.predict([[2016]]), color='red')
plt.plot(X_m, svr_m.predict(X_m), color='green')
plt.show()

#Print male prediction and stats
prediction = svr_m.predict([[2016]])[0]
actual = data_2016.loc[((data_2016['Event'] == '100 metres') 
                      & (data_2016['Gender'] == 'M')
                      & (data_2016['Position'] == 1)), 'Time'].values[0]

print('Correlation Coefficient: {:0.2f}'.format(svr_m.score(X_m, y_m)))
print('2016 Prediction: {:0.2f} seconds'.format(prediction))
print('2016 Actual Time: {0} seconds'.format(actual))
print('Error: {:0.3f}%'.format(1 - abs(prediction / actual)))


# In[ ]:


from sklearn.model_selection import GridSearchCV
#Tune SVR hyperparameters

def gridSearchFun (classifier, X_g, y_g):
    params = [{'kernel': ['rbf', 'linear', 'poly'],
               'degree': [1, 2, 3, 4],
               'gamma': ['scale', 'auto'] }]
    
    grid_search = GridSearchCV(estimator = classifier,
                              param_grid = params,
                              scoring = 'accuracy')
    grid_search.fit(X_g, y_g)
    best = grid_search.best_params_
    return str(best)

print (gridSearchFun(svr_f, X_f, y_f))


# In[ ]:





# In[ ]:





# In[ ]:




