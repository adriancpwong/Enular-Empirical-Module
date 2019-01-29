import tweepy 
from tweepy import OAuthHandler 
from tweepy import Cursor
from textblob import TextBlob 
import re
import csv
import datetime
import os 
import sys
import json
import time
import math
import pymongo
from pymongo import MongoClient
import numpy as np
import scipy
import sklearn
from sklearn import svm
from sklearn import tree
from sklearn import dummy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from iexfinance import Stock
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#stock to predict
predict = 'adobe'
predictStockCode = 'ADBE'

dateToday = datetime.date.today()
dateTodayStr = dateToday.strftime('%d-%m-%Y')

#initializes arrays
x_train_temp = []
y_train_temp = []
test_load = []

#converts to numpy arrays
x_train_load = np.asarray(x_train_temp)
y_train_load = np.asarray(y_train_temp)

#scalling datasets
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_load)
y_train_scaled = y_train_load.ravel()
test_scaled = scaler.fit_transform(test_load)

#splitting datasets for cross validation
x_train_pre, x_test_pre, y_train, y_test = train_test_split(x_train_scaled, y_train_scaled, test_size=0.2, random_state=0)

#reducing features
pca = PCA(n_components=2)
pca.fit(x_train_scaled)
x_train = pca.transform(x_train_pre)
x_test = pca.transform(x_test_pre)
test = pca.transform(test_scaled)

#f1 metrics
print("Metrics: F1, Accuracy, Precision, Recall")
metrics = tree.DecisionTreeClassifier(max_depth=10)
metrics.fit(x_train,y_train)
y_true = y_test
y_pred = metrics.predict(x_test)
f1metric = f1_score(y_true, y_pred, average='weighted')
accuracymetric = accuracy_score(y_true, y_pred)
precisionmetric = precision_score(y_true, y_pred, average='weighted')
recallmetric = recall_score(y_true, y_pred, average='weighted') 
print(f1metric)
print(accuracymetric)
print(precisionmetric)
print(recallmetric)

#bunch of shitty classifiers

dtc=tree.DecisionTreeClassifier(max_depth=10)
dtc.fit(x_train,y_train)
print ("DTC")
print(dtc.score(x_test,y_test))
	
smc=svm.SVC(gamma='scale',C=2.0)
smc.fit(x_train,y_train)
print ("SVM")
print(smc.score(x_test,y_test))

#knn=KNeighborsClassifier(n_neighbors=5,algorithm='auto',leaf_size=30,p=2)
#knn.fit(x_train,y_train)
#print ("KNN")
#print(knn.score(x_test,y_test))

gnb = GaussianNB()
gnb.fit(x_train,y_train)
print ("GNB")
print(gnb.score(x_test,y_test))

rfc = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
rfc.fit(x_train,y_train)
print ("RFC")
print(rfc.score(x_test,y_test))

etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(x_train,y_train)
print("ETC")
print(etc.score(x_test,y_test))

mlp = MLPClassifier(max_iter=1000,hidden_layer_sizes=(1000,))
mlp.fit(x_train,y_train)
print("MLP")
print(mlp.score(x_test,y_test))

dummy = DummyClassifier()
dummy.fit(x_train,y_train)
print("Dummy")
print(dummy.score(x_test,y_test))

#select best classifier
results = dtc.predict(test)

#optional save to csv
print(results)
#np.savetxt('results.csv', results, fmt='%d', delimiter=",", comments="")