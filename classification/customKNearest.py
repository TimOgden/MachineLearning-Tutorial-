import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

#3 points with the 'k' classification, 3 points with the 'r' classification
dataset= {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]


#Manual Euclidean Distance
#euclidean_dist = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups.')
	distances = []
	for group in data:
		for features in data[group]:
			#euclidean_dist = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
			euclidean_dist = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_dist, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

result = k_nearest_neighbors(dataset, new_features)
print(result)

for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0], ii[1], s=100, color=i)
plt.scatter(new_features[0],new_features[1], s=100, color=result)
plt.show()