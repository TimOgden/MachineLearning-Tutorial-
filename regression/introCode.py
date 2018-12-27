import pandas as pd
import quandl
import math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import numpy as np
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#Since this is just a linear regression, we need to change these features into something that the algorithm can better understand
#For example, a percentage of High/Low price would be a better feature than the two separate features.

#Using formula for percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

#Now let's overwrite our dataframe with a new dataframe of only what we care about
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'

df.fillna(-999999, inplace=True)

#This is the amount of days out we will predict using this data
forecast_out = int(math.ceil(0.01*len(df)))
#Need to shift the column up by the forecast_out amount because it is a future prediction
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


#1 means drop the column, not the row
X = np.array(df.drop(['label'],1))
#X = np.array(df.drop([forecast_col]), 1)
y = np.array(df['label'])
#Scikitlearn really helps in scaling
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)
clf = LinearRegression()
#fit is synonymous with train
clf.fit(X_train, y_train)
#score is synonymous with score
accuracy = clf.score(X_test, y_test)
print(df.head())
print("Accuracy of simple linear regression is {0:10.2f}%".format(accuracy*100))

clf2 = svm.SVR(kernel='linear')
clf2.fit(X_train, y_train)
accuracy = clf2.score(X_test, y_test)
print("Accuracy of support vector machine regression is {0:10.2f}%".format(accuracy*100))