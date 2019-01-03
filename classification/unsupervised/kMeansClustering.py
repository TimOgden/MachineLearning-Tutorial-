import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd
'''
X = np.array([[1,2],
				[1.5, 1.8],
				[5,8],
				[8,8],
				[1,0.6],
				[9,11]])
#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
plt.show()
'''


'''
Pclass Passenger class (1st,2nd,3rd)
survival (0 = no, 1 = yes)
name
sex
age
Number of siblings/spouses aboard
Number of parents/children aboard
ticket number
Passenger Fare (British pound)
Cabin
Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
Lifeboat
Body ID Number
Home/Destination
'''
#Now starting titanic clustering
df = pd.read_excel("titanic.xls")
df['male'] = (df['sex']=="male")
df['female'] = (df['sex']=="female")
df['cherbourg'] = (df['embarked']=="C")
df['queenstown'] = (df['embarked']=="Q")
df['southampton'] = (df['embarked']=="S")

df.drop(['body','name','sex','embarked'], 1, inplace=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
	columns = df.columns.values
	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x+=1
			df[column] = list(map(convert_to_int, df[column]))
	return df

df = handle_non_numerical_data(df)

df.drop(['ticket','boat'],1,inplace=True)
#print(df.head())
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
	predict_me = np.array(X[i].astype(float))
	predict_me = predict_me.reshape(-1, len(predict_me))
	prediction = clf.predict(predict_me)
	if prediction[0] == y[i]:
		correct+=1
if correct/len(X)>0.5:
	print(correct/len(X))
else:
	print(1-correct/len(X))