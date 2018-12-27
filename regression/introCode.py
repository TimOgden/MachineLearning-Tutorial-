import pandas as pd
import quandl
import math, datetime
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

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



#1 means drop the column, not the row
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#Scikitlearn really helps in scaling

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)

clf = LinearRegression()

try:
	pickle_in = open('linearregression.pickle', 'rb')
	clf = pickle.load(pickle_in)
	print("I successfully loaded in!")
except:
	#fit is synonymous with train
	clf.fit(X_train, y_train)
	#using pickle (python serialization) to save the classifier so it doesnt need training every time we run
	with open('linearregression.pickle', 'wb') as f:
		pickle.dump(clf, f)



#To load clf,
#pickle_in = open('linearregression.pickle', 'rb')
#clf = pickle.load(pickle_in)

#score is synonymous with score
accuracy = clf.score(X_test, y_test)
#print(df.head())
#print("Accuracy of simple linear regression is {0:10.2f}%".format(accuracy*100))

#clf2 = svm.SVR(kernel='linear')
#clf2.fit(X_train, y_train)
#accuracy2 = clf2.score(X_test, y_test)
#print("Accuracy of support vector machine regression is {0:10.2f}%".format(accuracy2*100))

#Make a prediction on all of the features that are from the forecast date onwards
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Stock Price')
plt.show()