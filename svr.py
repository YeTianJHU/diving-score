from sklearn.svm import SVR
import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir
from scipy.stats import spearmanr

feature_path = './p3d_features'
train_file = './training_idx.npy'
test_file = './testing_idx.npy'
label_file = './overall_scores.npy'

training_idx = np.load(train_file)
testing_idx = np.load(test_file)
label = np.load(label_file)[0]


x_train = []
y_train = []
for i in training_idx:
	i = int(i)
	feature_file = join(feature_path,"{:03d}.npy".format(i))
	feature = np.load(feature_file)
	feature = np.mean(feature, axis=0)
	x_train.append(feature)
	y_train.append(label[i-1])

x_test = []
y_test = []
for i in training_idx:
	i = int(i)
	feature_file = join(feature_path,"{:03d}.npy".format(i))
	feature = np.load(feature_file)
	feature = np.mean(feature, axis=0)
	x_test.append(feature)
	y_test.append(label[i-1])


# clf = SVR(C=9.1, epsilon=0.1)
# clf.fit(x_train, y_train)
# y_predit = clf.predict(x_test)

# rho, p_val = spearmanr(y_test, y_predit)
# print (rho)


# for i in range(len(y_test)):
# 	print (y_test[i],'-',y_predit[i])

c_range = [0.01, 0.1, 1, 10 ,100]
e_range = [0.01, 0.1, 1, 10 ,100]
k_range = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

for k in k_range:
	for i in c_range:
		   for j in e_range:

				   clf = SVR(C=i, epsilon=j,kernel=k)
				   clf.fit(x_train, y_train)
				   y_predit = clf.predict(x_test)

				   rho, p_val = spearmanr(y_test, y_predit)
				   print k_range, i, j, rho
	print ''