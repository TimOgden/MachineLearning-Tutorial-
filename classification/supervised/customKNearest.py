import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random




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
	confidence = Counter(votes).most_common(1)[0][1] / k
	return vote_result, confidence

accuracies = []


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
#print(df.head())
full_data = df.astype(float).values.tolist()
for i in range(25):
	random.shuffle(full_data)

	test_size = 0.2
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])
	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0
	confidence = 0
	for group in test_set:
		for data in test_set[group]:
			vote, confidence = k_nearest_neighbors(train_set, data, k=5)
			total+=1
			if group==vote:
				correct+=1
			#else:
				#print(confidence)

	#print('Accuracy: ', correct/total)
	#print('confidence: ', confidence)
	accuracies.append(correct/total)
print(sum(accuracies)/len(accuracies))