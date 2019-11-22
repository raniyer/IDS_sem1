#Data Preprocessing
#importing libraries
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

#importing dataset
dataset = pd.read_csv('master.csv')
dataset.drop(["country-year", "HDI for year"," gdp_for_year ($) ", "generation"], axis = 1, inplace = True)

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
Ndataset['country']= le.fit_transform(dataset['country']) 
dataset['sex']= le.fit_transform(dataset['sex'])
dataset['age']= le.fit_transform(dataset['age'])


# normalize the data attributes
normalized = preprocessing.normalize(dataset)


x1 = dataset.iloc[:, :-4] .values
x2 = dataset.iloc[:, -1:].values
X = np.concatenate((x1, x2), axis=1)
y = dataset.iloc[:, 6].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

xres = X[,3]
xres2 = X_train[:, 4]
plt.scatter(xres2, y_train, color = 'red')
plt.plot(xres2, regressor.predict(X_train), color = 'blue')
plt.title('GDP vs SUICIDE RATES(Training set)')
plt.show()


#Random Forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X, y)

a = np.array[[0,0.00635012,3.19583e-06,0,0.00254388]]
xnew = [[0],[0.00635012],[3.19583e-06],[0],[0.00254388]]
y_pred = regressor.predict([[0]])
# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(xres), max(xres), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(xres, y, color = 'red')
plt.plot(xres2, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()"""