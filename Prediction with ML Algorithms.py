# -*- coding: utf-8 -*-
"""
Created on Thursday May  2 12:18:59 2019

@author: aninyms
"""
#This project is on applying different machine learning algorithms to a breast cancer dataset to
#identify which model yeilds the highest prediction accuracy 

#importing the libraries needed 

import numpy as np 
import pandas as pd 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier


#ensuring missing values are recognized after reading in the file 
missing_values = ["n/a","na",'?']
column_names=['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'] 

#reading in the csv file 
data = pd.read_csv(r"insert filename and path here", na_values = missing_values,names=column_names, header = None)

#displaying the first couple of rows
data.head()

# preprocessing the data 
#dropping the BI-RADS column 
data1 = data.drop(columns="BI-RADS")

#checking to ensure the bi-rads column has been dropped 
data1.head(10)

#dealing with the missing values by using the median values in each column 
data1[data1.isnull().any(axis=1)]

data2=data1.fillna(data1.median())

data2.isnull()

# converting the severity column to 1 and 0
data2['Severity'] = data2['Severity'].map({'yes': 1, 'no': 0 })

data2.head()

#identifying the column we are predicting which is severity
target=['Severity']
not_target=['Age','Shape', 'Margin', 'Density']
 
#normalizing the data 
Y=data2[target]

normX=data2[not_target]

normX.head()
Y.head()

norm=Normalizer().fit(normX)
X = norm.transform(normX)

#splitting the dataset into training set and test set

x_train,x_test, y_train,y_test= train_test_split(X,Y,test_size=0.25, random_state=1)

#Decision tree & k-fold cross validation
dtree = DecisionTreeClassifier()

# training the model
dtree = dtree.fit(x_train,y_train)

#predicting the test dataset value
y_pred = dtree.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



#using cross_val_score
dtree=DecisionTreeClassifier()
treeresult= cross_val_score(dtree,X,Y, cv=10)
print(np.mean(treeresult))

#RANDOM FOREST
rf=RandomForestClassifier(n_estimators=10, max_depth=2)
rfcross= cross_val_score(rf,X,np.ravel(Y), cv=10)
print(np.mean(treeresult))


# KNN model 
#creating a for loop k to run values 1 to 50:
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_res= cross_val_score(knn,X,np.ravel(Y), cv=10)
    print("k:",k,"Acc:",np.mean(knn_res))


#Naive Bayes model
new1 = MultinomialNB()
bayes= cross_val_score(new1,X,np.ravel(Y), cv=10)
print(np.mean(treeresult))













