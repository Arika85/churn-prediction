import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('churndata.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#displaying data columns
print(X)
print(y)

# Encoding categorical data
#Label encoding 'Gender' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

#OneHot encoding location column and column transformer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#splitting the dataset to Train and Test
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Predicting Test results
y_pred = classifier.predict(X_test)

# Evaluating Metrics (confusion matrix)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# cross validating by applying K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(f"Accuracy: {accuracies.mean()*100}")
print(f"Standard Deviation: {accuracies.std()*100}")








     