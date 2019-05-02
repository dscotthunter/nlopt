import csv
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer

boston = load_boston()
X = boston.data 
Y = boston.target 
data = []
for num in range(X.shape[0]):
	entry = X[num,:].tolist()
	entry.append(Y[num])
	data.append(entry)
with open("data_as_csv/boston.csv", 'w') as csvfile:
	writer = csv.writer(csvfile)
	for row in data:
		writer.writerow(row)


breast_cancer = load_breast_cancer()
X = breast_cancer.data 
Y = breast_cancer.target 
print(X.shape)
print(Y.shape)
data = []
for num in range(X.shape[0]):
	entry = X[num,:].tolist()
	entry.append(Y[num])
	data.append(entry)
with open("data_as_csv/breast_cancer.csv", 'w') as csvfile:
	writer = csv.writer(csvfile)
	for row in data:
		writer.writerow(row)
