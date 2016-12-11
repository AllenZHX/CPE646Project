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

from sklearn.neighbors import KNeighborsClassifier
#for k in range(3,20,2):

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_data,train_labels)

results = []
for i in range(len(test_data)):
	res = neigh.predict(test_data[i])
	results.append(res)

correct = 0
wrong = 0
for j in range(len(test_labels)):
	if(results[j] == test_labels[j]):
		correct += 1
	else:
		wrong += 1
print("#####################################################################")
print("Correct rate is: " , float(correct)/(correct+wrong))
#(Correct rate is:  0.7303  when k=3)
#(Correct rate is:  0.7082  when k=5)
#(Correct rate is:  0.7603  when k=7)
#(Correct rate is:  0.7066  when k=9)
#(Correct rate is:  0.7082  when k=11)
#(Correct rate is:  0.7050  when k=13)
#(Correct rate is:  0.7145  when k=15)
#(Correct rate is:  0.7177  when k=17)
#(Correct rate is:  0.6782  when k=19)





