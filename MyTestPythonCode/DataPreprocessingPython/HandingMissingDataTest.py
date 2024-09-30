# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.impute import SimpleImputer
# Load the dataset
dataset = pd.read_csv("pima-indians-diabetes.csv")
# Identify missing data (assumes that missing data is represented as NaN)
print(dataset == 0)
print((dataset == 0).sum())
# Print the number of missing entries in each column
# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=0.0, strategy='mean')
# Fit the imputer on the DataFrame
imputer.fit(dataset)
# Apply the transform to the DataFrame
dataset = imputer.transform(dataset)
#Print your updated matrix of features
print(dataset)