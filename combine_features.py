
import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")


feature_path = '.tcn_p3d_features'
train_file = './data_files/training_idx.npy'
test_file = './data_files/testing_idx.npy'
# label_file = './data_files/difficulty_level.npy'
label_file = './data_files/overall_scores.npy'
uni_len = 9

training_idx = np.load(train_file)
testing_idx = np.load(test_file)
label = np.load(label_file)[0]

x_train_score = []
x_train_feature = []
y_train = []


for i in training_idx:
	i = int(i)
	feature_file1 = join(feature_path,"{:03d}_1.npy".format(i))
	feature_file2 = join(feature_path,"{:03d}_2.npy".format(i))
	feature_file3 = join(feature_path,"{:03d}_3.npy".format(i))
	feature_file4 = join(feature_path,"{:03d}_4.npy".format(i))
	feature1 = np.load(feature_file1)
	feature2 = np.load(feature_file2)
	feature3 = np.load(feature_file3)
	feature4 = np.load(feature_file4)

	score = [feature1[1],feature2[1],feature3[1],feature4[1]]

	feature = [feature1[0],feature2[0],feature3[0],feature4[0]]
	feature = np.mean(feature, axis=0)
	print feature.shape
	x_train_score.append(score)
	x_train_feature.append(feature)

	y_train.append(label[i-1])
	print score, label[i-1]
print len(x_train_score), len(y_train)
# print x_train_score[0].shape, y_train[0].shape
all_stuff = [x_train_feature,x_train_score,y_train]
np.save('./data_files/all_train_v2.npy',all_stuff)


x_test_score = []
x_test_feature = []
y_test = []
for i in testing_idx:
	i = int(i)
	feature_file1 = join(feature_path,"{:03d}_1.npy".format(i))
	feature_file2 = join(feature_path,"{:03d}_2.npy".format(i))
	feature_file3 = join(feature_path,"{:03d}_3.npy".format(i))
	feature_file4 = join(feature_path,"{:03d}_4.npy".format(i))
	feature1 = np.load(feature_file1)
	feature2 = np.load(feature_file2)
	feature3 = np.load(feature_file3)
	feature4 = np.load(feature_file4)

	score = [np.ndarray.tolist(feature1[1]),np.ndarray.tolist(feature2[1]),np.ndarray.tolist(feature3[1]),np.ndarray.tolist(feature4[1])]
	# score = extend(score, uni_len)
	score = [feature1[1],feature2[1],feature3[1],feature4[1]]

	feature = [feature1[0],feature2[0],feature3[0],feature4[0]]
	feature = np.mean(feature, axis=0)
	print feature.shape
	x_test_score.append(score)
	x_test_feature.append(feature)

	y_test.append(label[i-1])
	print score, label[i-1]
print len(x_test_score), len(y_test)
# print x_test_score[0].shape, y_test[0].shape
all_stuff = [x_test_feature,x_test_score,y_test]
np.save('./data_files/all_test_v2.npy',all_stuff)