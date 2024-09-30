# Importing the necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib as plt 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = pd.read_csv('titanic.csv')
print (dataset.head())
# Identify the categorical data
X = dataset.iloc[:,:]
Y = dataset.iloc[:,-1]
column_types = X.dtypes
print (column_types)

category_features = list(X.select_dtypes(include=['object']).columns)
#print (category_features)

category_features = ['Sex', 'Embarked', 'Pclass']
# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), category_features)], remainder='passthrough')
X = ct.fit_transform(X)
X = np.array(X)
print(X)
pd.DataFrame(X).to_csv('output_encoding_category1.csv', index=False)
print(X.shape)

X = dataset.iloc[:,:]
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Sex'])], remainder='passthrough')
X = ct.fit_transform(X)
X = np.array(X)
print (X)
pd.DataFrame(X).to_csv('output_encoding_category2.csv', index=False)
print(X.shape)
# Apply the fit_transform method on the instance of ColumnTransformer