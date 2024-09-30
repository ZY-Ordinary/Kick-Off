# Import necessary libraries
import numpy as np 
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wine Quality Red dataset
dataset = pd.read_csv('Winequlity-red.csv', delimiter=';')

# Separate features and target
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# Split the dataset into an 80-20 training-test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train)
# Create an instance of the StandardScaler class
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print(x_train)
print(x_test)

# Apply the transform to the test set


# Print the scaled training and test datasets