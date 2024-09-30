# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('.\\Part2Section5_MultipleRegression\\Python\\50_Startups.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print (x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set


# Predicting the Test set results
