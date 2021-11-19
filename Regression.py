#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# In[27]:


#Load data
data_all = pd.read_csv('Olympics_Athletics_Results_cleaned.csv')
data_all.drop(columns=['Unnamed: 0'], inplace=True)

#Display event and measurement options
print('Events to choose from: ', data_all.Event.unique().tolist())


# In[28]:


########## Choose event from above #############

event = '5,000 metres'

################################################


# In[29]:


#Specify type of event for proper measurement column
races = ('100 metres', '200 metres', '400 metres', '800 metres', 
          '1,500 metres', '5,000 metres', '10,000 metres', 'Marathon')

distances = ('Long Jump', 'Triple Jump', 'Shot Put', 'Discus Throw', 
             'Hammer Throw', 'Javelin Throw')

#Provide appropriate measurement based on type of event
if event in races:
    measurement = 'Time'
    unit = 'seconds'
elif event in distances:
    measurement = 'Distance' 
    unit = 'metres'
elif event in heights:
    measurement = 'Height'
    unit = 'metres'
else:
    print('Not a valid choice')
    
#Display event and metric
print('Event: ', event)
print('Measurement: ', measurement)
print('Unit: ', unit)

#Choose events with a none NaN measurement
data_no_nan = data_all[data_all[measurement].notnull()]


# In[30]:


#If marathon, convert time from seconds to decimal hours
#for more accurate modeling
if event == 'Marathon':
    data_no_nan.loc[(data_no_nan['Event'] == 'Marathon'), 'Time'] /= 3600.0


# In[31]:


#Split 2016 apart, only gold medalist, and create training and testing groups
data = data_no_nan[data_no_nan['Year'] != 2016]
data = data[data['Position'] == 1]

data_2016 = data_no_nan[data_no_nan['Year'] == 2016]
data = data[data['Position'] == 1]

#Separate by gender
data_f = data[data['Gender'] == 'F']
data_m = data[data['Gender'] == 'M']

#Create X and y
X_f = data_f.loc[(data_f['Event'] == event), ['Year']]
y_f = data_f.loc[(data_f['Event'] == event), [measurement]]

X_m = data_m.loc[(data_m['Event'] == event), ['Year']]
y_m = data_m.loc[(data_m['Event'] == event), [measurement]]


# In[32]:


#Create variable for 2016 result to graph and for comparison
actual_f = data_2016.loc[((data_2016['Event'] == event) 
                      & (data_2016['Gender'] == 'F')
                      & (data_2016['Position'] == 1)), measurement].values[0]

actual_m = data_2016.loc[((data_2016['Event'] == event) 
                      & (data_2016['Gender'] == 'M')
                      & (data_2016['Position'] == 1)), measurement].values[0]


# In[33]:


#Function for tuning using grid search
def gridSearchFun (reg, X, y, params):
    #Find the best hyperparameters
    grid_search = GridSearchCV(estimator = reg,
                              param_grid = params
                                )
    grid_search.fit(X, y.values.ravel())
    best = grid_search.best_params_
    return str(best)


# In[34]:


#Support Vector Regression

#Tuned below using GridSearchCV
svr_f = SVR(kernel='rbf')
svr_f.fit(X_f, y_f.values.ravel())

svr_m = SVR(kernel='rbf')
svr_m.fit(X_m, y_m.values.ravel())

#Graph female times
plt.title('Female {0} {1} in the Olympics (SVR)'.format(event, measurement))
plt.xlabel('Year')
plt.ylabel('{0} in {1}'.format(measurement, unit))
plt.scatter(X_f, y_f, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, svr_f.predict([[2016]]), color='green')
plt.plot(X_f, svr_f.predict(X_f), color='green')
plt.scatter(2016, actual_f, color='blue')
plt.show()

#Print female prediction and stats
prediction_f = svr_f.predict([[2016]])[0]

print('Correlation Coefficient: {:0.2f}'.format(svr_f.score(X_f, y_f)))
print('2016 Prediction: {:0.2f} {}'.format(prediction_f, unit))
print('2016 Actual {0}: {1} {2}'.format(measurement, actual_f, unit))
print('Error: {:0.1f}%'.format(abs(1 - prediction_f / actual_f) * 100))

#Graph male times
plt.title('Male {0} {1} in the Olympics (SVR)'.format(event, measurement))
plt.xlabel('Year')
plt.ylabel('{0} in {1}'.format(measurement, unit))
plt.scatter(X_m, y_m, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, svr_m.predict([[2016]]), color='green')
plt.plot(X_m, svr_m.predict(X_m), color='green')
plt.scatter(2016, actual_m, color='blue')
plt.show()

#Print male prediction and stats
prediction_m = svr_m.predict([[2016]])[0]

print('Correlation Coefficient: {:0.2f}'.format(svr_m.score(X_m, y_m)))
print('2016 Prediction: {:0.2f} {}'.format(prediction_m, unit))
print('2016 Actual {0}: {1} {2}'.format(measurement, actual_m, unit))
print('Error: {:0.1f}%'.format(abs(1 - (prediction_m / actual_m)) * 100))


# In[35]:


#Tune SVR hyperparameters

##### Choose gender ('F' or 'M') ######
gender = 'F'

if gender == 'F':
    regressor = svr_f
    X_g = X_f
    y_g = y_f

if gender == 'M':
    regressor = svr_m
    X_g = X_m
    y_g = y_m
    
parameters = [{'kernel': ['rbf', 'linear', 'poly'],
           'degree': [2, 3, 4]
         }]

#print ('Try these parameters: ', gridSearchFun(regressor, X_g, y_g, parameters))


# In[36]:


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures


#Make and fit the models
poly_f = PolynomialFeatures(degree = 4)
X_poly_f = poly_f.fit_transform(X_f)
poly_reg_f = LinearRegression()
poly_reg_f.fit(X_poly_f, y_f)

poly_m = PolynomialFeatures(degree = 4)
X_poly_m = poly_m.fit_transform(X_m)
poly_reg_m = LinearRegression()
poly_reg_m.fit(X_poly_m, y_m)

#Predictions
prediction_f = float(poly_reg_f.predict(poly_f.fit_transform([[2016]])))
prediction_m = float(poly_reg_m.predict(poly_m.fit_transform([[2016]])))

#Graph female times
plt.title('Female {0} {1} in the Olympics (Polynomial)'.format(event, measurement))
plt.xlabel('Year')
plt.ylabel('{0} in {1}'.format(measurement, unit))
plt.scatter(X_f, y_f, color='blue')
plt.xticks(np.arange(1900, 2024, 10))
plt.scatter(2016, prediction_f, color='green')
plt.plot(X_f, poly_reg_f.predict(X_poly_f), color='green')
plt.scatter(2016, actual_f, color='blue')
plt.show()

#Print female prediction and stats
print('Correlation Coefficient: {:0.2f}'.format(poly_reg_f.score(X_poly_f, y_f)))
print('2016 Prediction: {:0.2f} {}'.format(prediction_f, unit))
print('2016 Actual {0}: {1} {2}'.format(measurement, actual_f, unit))
print('Error: {:0.1f}%'.format(abs(1 - prediction_f / actual_f) * 100))

#Graph male times
plt.title('Male {0} {1} in the Olympics (Polynomial)'.format(event, measurement))
plt.xlabel('Year')
plt.ylabel('{0} in {1}'.format(measurement, unit))
plt.scatter(X_m, y_m, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, prediction_m, color='green')
plt.plot(X_m, poly_reg_m.predict(X_poly_m), color='green')
plt.scatter(2016, actual_m, color='blue')
plt.show()

#Print male prediction and stats
print('Correlation Coefficient: {:0.2f}'.format(poly_reg_m.score(X_poly_m, y_m)))
print('2016 Prediction: {:0.2f} {}'.format(prediction_m, unit))
print('2016 Actual {0}: {1} {2}'.format(measurement, actual_m, unit))
print('Error: {:0.1f}%'.format(abs(1 - prediction_m / actual_m) * 100))


# In[37]:


#Linear Regression

#Make and fit the models
line_reg_f = LinearRegression()
line_reg_f.fit(X_f, y_f)

line_reg_m = LinearRegression()
line_reg_m.fit(X_m, y_m)

#Graph female times
plt.title('Female {0} {1} in the Olympics (Linear)'.format(event, measurement))
plt.xlabel('Year')
plt.ylabel('{0} in {1}'.format(measurement, unit))
plt.scatter(X_f, y_f, color='blue')
plt.xticks(np.arange(1900, 2024, 10))
plt.scatter(2016, line_reg_f.predict([[2016]]), color='green')
plt.plot(X_f, line_reg_f.predict(X_f), color='green')
plt.scatter(2016, actual_f, color='blue')
plt.show()

#Print female prediction and stats
prediction_f = line_reg_f.predict([[2016]])[0][0]

print('Correlation Coefficient: {:0.2f}'.format(line_reg_f.score(X_f, y_f)))
print('2016 Prediction: {:0.2f} {}'.format(prediction_f, unit))
print('2016 Actual {0}: {1} {2}'.format(measurement, actual_f, unit))
print('Error: {:0.1f}%'.format(abs(1 - prediction_f / actual_f) * 100))

#Graph male times
plt.title('Male {0} {1} in the Olympics (Linear)'.format(event, measurement))
plt.xlabel('Year')
plt.ylabel('{0} in {1}'.format(measurement, unit))
plt.scatter(X_m, y_m, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, line_reg_m.predict([[2016]]), color='green')
plt.plot(X_m, line_reg_m.predict(X_m), color='green')
plt.scatter(2016, actual_m, color='blue')
plt.show()

#Print male prediction and stats
prediction_m = line_reg_m.predict([[2016]])[0][0]

print('Correlation Coefficient: {:0.2f}'.format(line_reg_m.score(X_m, y_m)))
print('2016 Prediction: {:0.2f} {}'.format(prediction_m, unit))
print('2016 Actual {0}: {1} {2}'.format(measurement, actual_m, unit))
print('Error: {:0.1f}%'.format(abs(1 - prediction_m / actual_m) * 100))


# In[38]:


#Decision Tree Regression


#Fit the model
#Tuned using GridSearchCV below
dtr_f = DecisionTreeRegressor(criterion='friedman_mse', splitter='random', random_state=42)
dtr_m = DecisionTreeRegressor(criterion='friedman_mse', splitter='best', random_state=42)

dtr_f.fit(X_f, y_f)
dtr_m.fit(X_m, y_m)

#Graph female times
plt.title('Female {0} {1} in the Olympics (Decision Tree)'.format(event, measurement))
plt.xlabel('Year')
plt.ylabel('{0} in {1}'.format(measurement, unit))
plt.scatter(X_f, y_f, color='blue')
plt.xticks(np.arange(1920, 2024, 10))
plt.scatter(2016, dtr_f.predict([[2016]]), color='green')
plt.plot(X_f, dtr_f.predict(X_f), color='green')
plt.scatter(2016, actual_f, color='blue')
plt.show()

#Print female prediction and stats
prediction_f = dtr_f.predict([[2016]])[0]

print('Correlation Coefficient: {:0.2f}'.format(dtr_f.score(X_f, y_f)))
print('2016 Prediction: {:0.2f} {}'.format(prediction_f, unit))
print('2016 Actual {0}: {1} {2}'.format(measurement, actual_f, unit))
print('Error: {:0.1f}%'.format(abs(1 - prediction_f / actual_f) * 100))

#Graph male times
plt.title('Male {0} {1} in the Olympics (Decision Tree)'.format(event, measurement))
plt.xlabel('Year')
plt.ylabel('{0} in {1}'.format(measurement, unit))
plt.scatter(X_m, y_m, color='blue')
plt.xticks(np.arange(1890, 2024, 10))
plt.scatter(2016, dtr_m.predict([[2016]]), color='green')
plt.plot(X_m, dtr_m.predict(X_m), color='green')
plt.scatter(2016, actual_m, color='blue')
plt.show()

#Print male prediction and stats
prediction_m = dtr_m.predict([[2016]])[0]

print('Correlation Coefficient: {:0.2f}'.format(dtr_m.score(X_m, y_m)))
print('2016 Prediction: {:0.2f} {}'.format(prediction_m, unit))
print('2016 Actual {0}: {1} {2}'.format(measurement, actual_m, unit))
print('Error: {:0.1f}%'.format(abs(1 - prediction_m / actual_m) * 100))


# In[39]:


#Tune decision tree hyperparameters

##### Choose gender ('F' or 'M') ######
gender = 'F'

if gender == 'F':
    regressor = dtr_f
    X_g = X_f
    y_g = y_f

if gender == 'M':
    regressor = dtr_m
    X_g = X_m
    y_g = y_m

params = [{'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
           'splitter': ['best', 'random']
          }]

#print ('Try these parameters: ', gridSearchFun(regressor, X_g, y_g, parameters))


# In[40]:


#Print all scores
print(event, 'Regression Correlation Coefficients')
#Female
print('\nFemale:')
print(' SVR: {:0.2f}'.format(svr_f.score(X_f, y_f)))
print(' Poly: {:0.2f}'.format(poly_reg_f.score(X_poly_f, y_f)))
print(' Linear: {:0.2f}'.format(line_reg_f.score(X_f, y_f)))
print(' DT: {:0.2f}'.format(dtr_f.score(X_f, y_f)))

#Male
print('\nMale:')
print(' SVR: {:0.2f}'.format(svr_m.score(X_m, y_m)))
print(' Poly: {:0.2f}'.format(poly_reg_m.score(X_poly_m, y_m)))
print(' Linear: {:0.2f}'.format(line_reg_m.score(X_m, y_m)))
print(' DT: {:0.2f}'.format(dtr_m.score(X_m, y_m)))


# In[ ]:




