import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

file=pd.read_csv("creditcard.csv")

file.head(10)

file.describe()

file.isnull().sum()

file['Class'].value_counts()

normal=file[file.Class==0]

fraud=file[file.Class==1]

print(normal.shape)

print(normal.shape)

normal.Amount.describe()

fraud.Amount.describe()

file.groupby('Class').mean()

normal_sample=normal.sample(n=492)

new_file=pd.concat([normal_sample,fraud],axis=0)

new_file.head(10)

new_file['Class'].value_counts()

new_file.groupby('Class').mean()

X=new_file.drop(columns='Class',axis=1)

Y=new_file['Class']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

model=LogisticRegression()

model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)

training_data_acuracy=accuracy_score(X_train_prediction,Y_train)*100

print(f"Training Data Accuracy: {training_data_acuracy}%")

X_test_prediction=model.predict(X_test)

test_data_accuracy=accuracy_score(X_test_prediction,Y_test)*100

print(f"Test Data Accuracy: {test_data_accuracy}%")
