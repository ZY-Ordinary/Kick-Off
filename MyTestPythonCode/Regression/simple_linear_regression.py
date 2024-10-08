# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Create a figure instance with ax1 and ax2
fig, (ax1, ax2)= plt.subplots(1,2)

# Visualising the Training set results
ax1.scatter(X_train, y_train, color = 'red')
ax1.plot(X_train, regressor.predict(X_train), color = 'blue')
ax1.set_title('Salary vs Experience (Training set)')
ax1.set_xlabel('Years of Experience')
ax1.set_ylabel('Salary')

# Visualising the Test set results
ax2.scatter(X_test, y_test, color = 'red')
ax2.plot(X_train, regressor.predict(X_train), color = 'blue')
ax2.set_title('Salary vs Experience (Test set)')
ax2.set_xlabel('Years of Experience')
ax2.set_ylabel('Salary')

plt.show()

# Visualising the Training set results
#plt.scatter(X_train, y_train, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Salary vs Experience (Training set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show()

# Visualising the Test set results
#plt.scatter(X_test, y_test, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Salary vs Experience (Test set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show()