import datetime
import os 
import sys
import json
import time
import math
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
import selenium

print ("zac is a faggot")