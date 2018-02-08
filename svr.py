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
train_file = './training_idx.npy'
test_file = './testing_idx.npy'
# label_file = './difficulty_level.npy'
label_file = './overall_scores.npy'
uni_len = 9


training_idx = np.load(train_file)
testing_idx = np.load(test_file)
label = np.load(label_file)[0]

def extend(score, uni_len):
	if len(score) > uni_len:
		score = score[len(score)-uni_len:]
	elif len(score) < uni_len:
		score = np.insert(score, 0, np.full((uni_len-len(score)), score[0]))
	return score

# x_train = []
# y_train = []
# for i in training_idx:
# 	i = int(i)
# 	feature_file = join(feature_path,"{:03d}.npy".format(i))
# 	feature = np.load(feature_file)
# 	score = feature[:, 1]
# 	score = extend(score, uni_len)
# 	# print score.shape
# 	x_train.append(score)
# 	y_train.append(label[i-1])
# 	# print (score, label[i-1])
# print len(x_train), len(y_train)
# print x_train[0].shape, y_train[0].shape

# x_test = []
# y_test = []
# for i in testing_idx:
# 	i = int(i)
# 	feature_file = join(feature_path,"{:03d}.npy".format(i))
# 	feature = np.load(feature_file)
# 	score = feature[:, 1]
# 	score = extend(score, uni_len)
# 	x_test.append(score)
# 	y_test.append(label[i-1])
# print len(x_test), len(y_test)
# print x_test[0].shape, y_test[0].shape

x_train = []
y_train = []
for i in training_idx:
	i = int(i)
	feature_file = join(feature_path,"{:03d}.npy".format(i))
	feature = np.load(feature_file)
	score = feature[:, 0]
	score = np.mean(score, axis=0)
	# score = score/np.linalg.norm(score)
	# print score.shape
	x_train.append(score)
	y_train.append(label[i-1])
	# print (score, label[i-1])
print len(x_train), len(y_train)
print x_train[0].shape, y_train[0].shape

x_test = []
y_test = []
for i in testing_idx:
	i = int(i)
	feature_file = join(feature_path,"{:03d}.npy".format(i))
	feature = np.load(feature_file)
	score = feature[:, 0]
	score = np.mean(score, axis=0)
	# score = score/np.linalg.norm(score)
	x_test.append(score)
	y_test.append(label[i-1])
print len(x_test), len(y_test)
print x_test[0].shape, y_test[0].shape


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

# c_range = [0.01, 0.1, 1, 10 ,100]
# e_range = [0.01, 0.1, 1, 10]
# k_range = ['rbf'] #'linear', 'poly', 

# for k in k_range:
# 	for i in c_range:
# 		for j in e_range:

# 			clf = SVR(C=i, epsilon=j,kernel=k)
# 			clf.fit(x_train, y_train)
# 			y_predit = clf.predict(x_test)

# 			rho, p_val = spearmanr(y_test, y_predit)
# 			#print k_range, i, j, rho
# 			logging.info("k:{0}, C:{1}, e:{2}, rho:{3}".format(k, i, j, rho))
# 			print (k, i, j, rho)


clf = LinearRegression()
clf.fit(x_train, y_train)
y_predit = clf.predict(x_test)

rho, p_val = spearmanr(y_test, y_predit)
#print k_range, i, j, rho
# logging.info("k:{0}, C:{1}, e:{2}, rho:{3}".format(k, i, j, rho))
print (rho)
# logging.info("{0}".format(rho))