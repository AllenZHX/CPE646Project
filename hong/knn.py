import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

all_data = pd.read_csv('../data/voice.csv')
label = all_data.pop('label')

all_data = all_data.values

label.replace(['male','female'], [1, 0], inplace = True)
label = label.values

train_data, test_data, train_labels, test_labels = train_test_split(all_data, label, test_size = 0.2)

train_data, test_data, train_labels, test_labels = np.array(train_data, dtype = 'float32'), np.array(test_data, dtype = 'float32'),np.array(train_labels, dtype = 'float32'),np.array(test_labels, dtype = 'float32')

print(len(train_data))
print(len(test_data))

import math
def calculateDistance(instance1, instance2,length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]),2)
	return math.sqrt(distance)

import operator
def getNeighbors(train_data, train_labels, test_data_single, k):
	distances = []
	length = 20
	for x in range(len(train_data)):
		dist = calculateDistance(test_data_single, train_data[x],length)
		distances.append((train_data[x],train_labels[x],dist))
	distances.sort(key=operator.itemgetter(2))  #sort based on 3rd item which is dist
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][1])  #return top k nearest data's label
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

#####################  main  #####################
#for tt in range(3,18,2):
for tt in range(21,52,10):
	K=tt
	results = []
	for i in range(len(test_data)):
		x = test_data[i]
		neighbor = getNeighbors(train_data, train_labels, x, K)
		res = getResponse(neighbor)
		results.append(res)

	correct = 0
	wrong = 0
	for j in range(len(test_labels)):
		if(results[j] == test_labels[j]):
			correct += 1
		else:
			wrong += 1
	print("Correct rate is: " , float(correct)/(correct+wrong))
# the result is about 0.7145   when K=3
# the result is about 0.7161   when K=5
# the result is about 0.6719   when K=7
# the result is about 0.6861   when K=9
# the result is about 0.6845   when K=11
# the result is about 0.6845   when K=13
# the result is about 0.6877   when K=15
# the result is about 0.6798   when K=17
# the result is about 0.6972   when K=21
# the result is about 0.6830   when K=31
# the result is about 0.6893   when K=41
# the result is about 0.6924   when K=51

















