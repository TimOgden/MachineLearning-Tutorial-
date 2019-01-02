import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],
				[1.5, 1.8],
				[5,8],
				[8,8],
				[1,0.6],
				[9,11]])
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g","r","c","b","k","o"]