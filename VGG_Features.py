#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from keras.models import model_from_json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import shutil
import keras.backend as K

import subprocess
import keras
from keras.models import Sequential, Model 
from keras.layers import Input, merge, Dense, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dropout
from keras.regularizers import l2
# from keras.regularizers import ActivityRegularizer
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras import backend as K

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

dim_ordering = K.image_dim_ordering()

backend = dim_ordering


data_dir = './frames'
feature_dir = './resFeatures'


def main():


# Load in the pre-trained model

	# model = keras.applications.VGG19(include_top=False,weights='imagenet')

	model = ResNet50(include_top=True,weights='imagenet')

	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


	if not os.path.isdir(feature_dir):
		os.mkdir(feature_dir)

	fold_names = [f for f in os.listdir(data_dir)]
	n = 0
	for video_id, video_dir in enumerate(fold_names):
		n += 1
		feature_name = video_dir+'.npy'
		# print (feature_name)

		start = time.time()
		video_path = os.path.join(data_dir, video_dir)
		sort_img = np.sort(os.listdir(video_path))
		
		video_feature = []
		for imgs in sort_img:
			img = image.load_img(os.path.join(video_path,imgs), target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			features = model.predict(x)
			# print (features.shape)
			video_feature.append(features)
		print (feature_name,len(video_feature))

		np.save(os.path.join(feature_dir,feature_name),video_feature)



if __name__ == '__main__':
	main()