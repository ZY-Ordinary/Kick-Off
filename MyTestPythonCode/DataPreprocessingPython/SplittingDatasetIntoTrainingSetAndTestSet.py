# Import necessary libraries
import numpy as np
import pandas as pd 
import matplotlib as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
dataset = pd.read_csv('iris.csv')
# Separate features and target
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
# Split the dataset into an 80-20 training-test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print (x_train)
print (type(x_train))
print (y_train)
print (type(y_train))
print (x_test)
print (type(x_test))
print (y_test)
print (type(y_test))
# Apply feature scaling on the training and test sets
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Print the scaled training and test sets
print (x_train)
print (type(x_train))
print (x_test)
print (type(x_test))