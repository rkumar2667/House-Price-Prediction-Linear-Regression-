#!/usr/bin/env python
# coding: utf-8


#import dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

#load dataset
dataset = pd.read_csv('housing.csv')
dataset.head()


#check wheather there are any missing values or null
dataset.isnull().sum()

#impute the missing values
total_bedroms = dataset[dataset["total_bedrooms"].notnull()]["total_bedrooms"]#["total_bedrooms"]
imputer = SimpleImputer(np.nan,strategy ="median")
imputer.fit(dataset.iloc[:,4:5])
dataset.iloc[:,4:5] = imputer.transform(dataset.iloc[:,4:5])
dataset.isnull().sum()


# Label encode for categorical feature (ocean_proximity)

labelEncoder = preprocessing.LabelEncoder()
print(dataset["ocean_proximity"].value_counts())
dataset["ocean_proximity"] = labelEncoder.fit_transform(dataset["ocean_proximity"])
dataset["ocean_proximity"].value_counts()
dataset.describe()


#transform data into dataframe
#data = data we want or the indepedent variables also known as x
#feature_names = column names of the data
#target = the target variable or the price of the houses or the dependent variable / y
dataset_x = dataset.drop("median_house_value",axis=1)
#dataset_x.head()
dataset_y = dataset["median_house_value"]
#dataset_y.head()


#Split into training and test set data (80% training & 20%testing)
X_train,X_test,y_train,y_test = train_test_split(dataset_x,dataset_y,test_size=0.2,random_state=42)


#normalzing input daya
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


#initialize linear regression model
reg = linear_model.LinearRegression() 


#training the data with our training data
reg.fit(X_train,y_train)
reg.score(X_train,y_train)
#64% training accuracy 

#print the coeffient/weights for each feature/column of our model
print(reg.coef_) #f(x) = y = mx + c #finding m here


#the prediction on our test data
y_pred = reg.predict(X_test)
print(y_pred)


#Check the performance of the model using mean squared error (MSE) 
print(mean_squared_error(y_test,y_pred))

