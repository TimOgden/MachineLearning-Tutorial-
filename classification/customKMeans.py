import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn import preprocessing
X = np.array([[1,2],
				[1.5, 1.8],
				[5,8],
				[8,8],
				[1,0.6],
				[9,11]])
#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

#clf = KMeans(n_clusters=2)
#clf.fit(X)
#centroids = clf.cluster_centers_
#labels = clf.labels_

colors = ["g","r","c","b","k","o"]

class K_Means:
	def __init__(self, k=2, tol=0.001, max_iter=300, disp_iter=1):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter
		self.disp_iter = disp_iter

	def fit(self, data):
		self.centroids = {}
		for i in range(self.k):
			self.centroids[i] = data[i]
		for i in range(self.max_iter):
			self.classifications = {}
			for i in range(self.k):
				self.classifications[i] = []
			for featureset in data:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)
			prev_centroids = dict(self.centroids)
			for classification in self.classifications:
				#pass
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)
			optimized = True
			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid-original_centroid)/original_centroid*100) > self.tol:
					optimized = False
			if self.disp_iter != -1 and i%self.disp_iter==0:
				self.plot_data(self.centroids, self.classifications)
			if optimized:
				break

	def predict(self, data):
		distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

	def plot_data(self, centroids, data):
		for centroid in centroids:
			plt.scatter(centroids[centroid][0], centroids[centroid][1], 
				marker="o", color="k", s=150, linewidths=5)
		for classification in data:
			color = colors[classification]
			for featureset in data[classification]:
				plt.scatter(featureset[0], featureset[1], 
					marker="x",color=color, s=150, linewidths=5)
		plt.show()



#unknowns = np.array([[1,3],
#					[8,9],
#					[0,3],
#					[5,4],
#					[6,4]])
#print(clf.centroids)
#for centroid in clf.centroids:
#	plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1], marker="*", color="k", s=100, linewidths=5)
#for unknown in unknowns:
#	classification = clf.predict(unknown)
#	plt.scatter(unknown[0],unknown[1], marker="*",
#		color=colors[classification], s=150, linewidths=5)
#plt.show()

#Let's test the titanic dataset on our custom K Means Algorithm
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

clf = K_Means(disp_iter=-1)
clf.fit(X)
correct = 0
for c, answer in enumerate(y):
	if answer == clf.predict(X[c]):
		correct+=1
acc = correct/len(y)
if acc < .5:
	acc = 1-acc
print('Accuracy:', acc)