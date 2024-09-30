# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


# Loading the Iris dataset
dataset = pd.read_csv("iris.csv")
print(dataset.head())
# Creating the matrix of features (X) and the dependent variable vector (y)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# Printing the matrix of features and the dependent variable vectorS
print (X)
print (y)