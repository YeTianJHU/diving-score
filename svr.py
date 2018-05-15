from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")


feature_path = './p3d_features'
train_file = './data_files/training_idx.npy'
test_file = './data_files/testing_idx.npy'
# label_file = './data_files/difficulty_level.npy'
label_file = './data_files/overall_scores.npy'
train_file = './data_files/all_train_v2.npy'
test_file = './data_files/all_test_v2.npy'


def to_array(l):
	output = np.zeros_like(len(l))
	for i in l:
		print i
		output = np.append(output, i)
	return output[1:]


train_data = np.array(np.load(train_file))
test_data = np.array(np.load(test_file))

x_train = np.array(train_data[0])
x_test = np.array(test_data[0])

print train_data.shape

# if using the concated feature, uncomment this
# x_train = np.array(train_data[-1])
# x_test = np.array(test_data[-1])

y_train = np.array(train_data[2])
y_test = np.array(test_data[2])


for idx, i in enumerate(x_train):
	for idy, j in enumerate(i):
		x_train[idx][idy] = round(float(j),1)

x_train = np.array(list(x_train), dtype=np.float)
y_train = np.array(y_train, dtype=np.float)

for idx, i in enumerate(x_test):
	for idy, j in enumerate(i):
		x_test[idx][idy] = round(float(j),1)

x_test = np.array(list(x_test), dtype=np.float)
y_test = np.array(y_test, dtype=np.float)

# for i in range(len(x_train)):
# 	print x_train[i], y_train[i]
print len(x_train), len(y_train)

clf = LinearRegression()
clf.fit(x_train, y_train)
y_predit = clf.predict(x_test)

rho, p_val = spearmanr(y_test, y_predit)
print rho

clf = SVR(C=221, epsilon=0.01)
clf.fit(x_train, y_train)
y_predit = clf.predict(x_test)

rho, p_val = spearmanr(y_test, y_predit)
print (rho)

clf = SVR(C=100, epsilon=0.001)
clf.fit(x_train, y_train)
y_predit = clf.predict(x_test)

rho, p_val = spearmanr(y_test, y_predit)
print (rho)

# for i in range(len(y_test)):
# 	print (y_test[i],'-',y_predit[i])

c_range = [60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260]
e_range = [7, 8,9,10,11,12,13,14]
k_range = ['rbf', 'linear']# 'poly', 

for k in k_range:
	for i in c_range:
		for j in e_range:

			clf = SVR(C=i, epsilon=j,kernel=k)
			clf.fit(x_train, y_train)
			y_predit = clf.predict(x_test)

			rho, p_val = spearmanr(y_test, y_predit)
			print k, i, j, round(rho, 3)
			# logging.info("k:{0}, C:{1}, e:{2}, rho:{3}".format(k, i, j, rho))
			# print (k, i, j, rho)


