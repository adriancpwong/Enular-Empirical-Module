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

client = MongoClient()
db = client.database_one
collection = db.collection_one
posts = db.posts

#for date in dates
#   for company in companies
#       deconstruct features to temp 2d array
#       aggregate to 1d array
#       add to xtrain 2d array
#       add stock price to ytrain 1d array

predict = 'adobe'

companyNameOne = 'apple'
companyNameTwo = 'intel'
companyNameThree = 'amazon'
companyNameFour = 'microsoft'
companyNameFive = 'alibaba'
companyNameSix = 'facebook'
companyNameSeven = 'alphabet'
companyNameEight = 'disney'
companyNameNine = 'micron'
companyNameTen = 'netflix'

companies = [companyNameOne,companyNameTwo,companyNameThree,companyNameFour,companyNameFive,companyNameSix,companyNameSeven,companyNameEight,companyNameNine,companyNameTen]
dates = ["07-01-2019","08-01-2019","09-01-2019","10-01-2019","11-01-2019","14-01-2019","15-01-2019","16-01-2019","17-01-2019","18-01-2019","21-01-2019","22-01-2019"]

dateToday = datetime.date.today()
dateTodayStr = dateToday.strftime('%d-%m-%Y')

x_train_temp = []
y_train_temp = []
test_load = []

feed = 0
for date in dates:
    for company in companies:

        aggregation = [float(0.00)]

        for post in posts.find({"Company":company,"DateStr":date,"Predict":0}):
            change = post['Change']
            aggregation.append(float(change))
        
        npagg = np.asarray(aggregation)
        y_train_temp.append(math.ceil(np.mean(npagg)))
        feed += 1

print(feed)
print(y_train_temp)

for date in dates:
    for company in companies:

        followsArray = [float(0.00)]
        friendsArray = [float(0.00)]
        listsArray = [float(0.00)]
        #userfavouritesArray = [float(0.00)]
        statusesArray = [float(0.00)]
        quotesArray = [float(0.00)]
        repliesArray = [float(0.00)]
        retweetsArray = [float(0.00)]
        favouritesArray = [float(0.00)]
        sentimentArray = [float(0.00)]
        lengthArray = [float(0.00)]
        mentionsArray = [float(0.00)]
        hashesArray = [float(0.00)]
        atsArray = [float(0.00)]
        wordArray = [float(0.00)]

        tempArray = []

        for post in posts.find({"Company":company,"DateStr":date,"Predict":0}):

            followsArray.append(float(post['Followers']))
            friendsArray.append(float(post['Friends']))
            listsArray.append(float(post['Lists']))
            #userfavouritesArray.append(float('0'))
            statusesArray.append(float(post['Statuses']))
            quotesArray.append(float(post['Quotes']))
            repliesArray.append(float(post['Replies']))
            retweetsArray.append(float(post['Retweets']))
            favouritesArray.append(float(post['Favourites']))
            sentimentArray.append(float(post['Sentiment']))
            lengthArray.append(float(post['Length']))
            mentionsArray.append(float(post['Mentions']))
            hashesArray.append(float(post['Hashes']))
            atsArray.append(float(post['Ats']))
            wordArray.append(float(post['Words']))

        tempArray.append(np.mean(np.asarray(followsArray)))
        tempArray.append(np.mean(np.asarray(friendsArray)))
        tempArray.append(np.mean(np.asarray(listsArray)))
        #tempArray.append(np.mean(np.asarray(userfavouritesArray)))
        tempArray.append(np.mean(np.asarray(statusesArray)))
        tempArray.append(np.mean(np.asarray(quotesArray)))
        tempArray.append(np.mean(np.asarray(repliesArray)))
        tempArray.append(np.mean(np.asarray(retweetsArray)))
        tempArray.append(np.mean(np.asarray(favouritesArray)))
        tempArray.append(np.mean(np.asarray(sentimentArray)))
        tempArray.append(np.mean(np.asarray(lengthArray)))
        tempArray.append(np.mean(np.asarray(mentionsArray)))
        tempArray.append(np.mean(np.asarray(hashesArray)))
        tempArray.append(np.mean(np.asarray(atsArray)))
        tempArray.append(np.mean(np.asarray(wordArray)))

        x_train_temp.append(tempArray)

print(x_train_temp)

if predict != '':

    feeds = 0

    followsArray = [float(0.00)]
    friendsArray = [float(0.00)]
    listsArray = [float(0.00)]
    #userfavouritesArray = [float(0.00)]
    statusesArray = [float(0.00)]
    quotesArray = [float(0.00)]
    repliesArray = [float(0.00)]
    retweetsArray = [float(0.00)]
    favouritesArray = [float(0.00)]
    sentimentArray = [float(0.00)]
    lengthArray = [float(0.00)]
    mentionsArray = [float(0.00)]
    hashesArray = [float(0.00)]
    atsArray = [float(0.00)]
    wordArray = [float(0.00)]

    tempArray = []

    for post in posts.find({"Company":predict,"DateStr":dateTodayStr,"Predict":1}):

        followsArray.append(float(post['Followers']))
        friendsArray.append(float(post['Friends']))
        listsArray.append(float(post['Lists']))
        #userfavouritesArray.append(float(0))
        statusesArray.append(float(post['Statuses']))
        quotesArray.append(float(post['Quotes']))
        repliesArray.append(float(post['Replies']))
        retweetsArray.append(float(post['Retweets']))
        favouritesArray.append(float(post['Favourites']))
        sentimentArray.append(float(post['Sentiment']))
        lengthArray.append(float(post['Length']))
        mentionsArray.append(float(post['Mentions']))
        hashesArray.append(float(post['Hashes']))
        atsArray.append(float(post['Ats']))
        wordArray.append(float(post['Words']))

    tempArray.append(np.mean(np.asarray(followsArray)))
    tempArray.append(np.mean(np.asarray(friendsArray)))
    tempArray.append(np.mean(np.asarray(listsArray)))
    #tempArray.append(np.mean(np.asarray(userfavouritesArray)))
    tempArray.append(np.mean(np.asarray(statusesArray)))
    tempArray.append(np.mean(np.asarray(quotesArray)))
    tempArray.append(np.mean(np.asarray(repliesArray)))
    tempArray.append(np.mean(np.asarray(retweetsArray)))
    tempArray.append(np.mean(np.asarray(favouritesArray)))
    tempArray.append(np.mean(np.asarray(sentimentArray)))
    tempArray.append(np.mean(np.asarray(lengthArray)))
    tempArray.append(np.mean(np.asarray(mentionsArray)))
    tempArray.append(np.mean(np.asarray(hashesArray)))
    tempArray.append(np.mean(np.asarray(atsArray)))
    tempArray.append(np.mean(np.asarray(wordArray)))

    test_load.append(tempArray)
else:
    test_load = [[50,50]]

print(feeds)
print(test_load)
#loads files

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

results = dtc.predict(test)

print(results)
#np.savetxt('results.csv', results, fmt='%d', delimiter=",", comments="")