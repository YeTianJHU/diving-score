import os
from os import listdir
from os.path import isfile, join, isdir

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

#import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torchvision.transforms import ToPILImage

import csv
# import skvideo.io

class divingDataset(Dataset):
	def __init__(self, data_folder, data_file, label_file, transform, random=False, test=False, num_frame=16, channel=3, size=160):

		self.data_folder = data_folder
		self.transform = transform
		self.num_frame = num_frame
		self.channel = channel
		self.size = size
		self.video_name = np.load(data_file)
		self.label = np.load(label_file)
		self.random = random
		self.test = test


		
	def __getitem__(self, index):

		video_name = str(self.video_name[index][0]).zfill(3) 
		# print video_name
		video_path = os.path.join(self.data_folder, video_name)

		if self.test:
			print 'test', self.test
			video_tensor, num_tensor = self.get_test_tensor(video_path, self.num_frame, self.channel, self.size)
			labels = self.label[0][self.video_name[index][0]-1].astype(np.float32)

			return video_tensor, num_tensor, labels
		else:
			video_tensor = self.get_video_tensor(video_path, self.num_frame, self.channel, self.size, self.random)

			labels = self.label[0][self.video_name[index][0]-1].astype(np.float32)

			return video_tensor, labels

	def __len__(self):
		return len(self.video_name)

	def collect_files(self, dir_name, file_ext=".jpg", sort_files=True):
		allfiles = [os.path.join(dir_name,f) for f in listdir(dir_name) if isfile(join(dir_name,f))]

		these_files = []
		for i in range(0,len(allfiles)):
			_, ext = os.path.splitext(os.path.basename(allfiles[i]))
			if ext == file_ext:
				these_files.append(allfiles[i])

		if sort_files and len(these_files) > 0:
			these_files = sorted(these_files)

		return these_files

	def get_video_tensor(self, dir, num_frame, channel, size, random):
		images = self.collect_files(dir)
		flow = torch.FloatTensor(channel,num_frame,size,size)
		if random==True:
			seed = np.random.random_integers(0,len(images)-num_frame) #random sampling
			for i in range(num_frame):
				img = Image.open(images[i+seed])
				img = img.convert('RGB')
				img = self.transform(img)
				flow[:,i,:,:] = img
		else: 
			downsampe = []
			for i in range(0, len(images), int(len(images)/num_frame)):
				downsampe.append(i)
			downsampe = downsampe[len(downsampe)-num_frame:]

			for idx, i in enumerate(downsampe):
				img = Image.open(images[i])
				img = img.convert('RGB')
				img = self.transform(img)
				flow[:,idx,:,:] = img
		return flow

	def get_test_tensor(self, dir, num_frame, channel, size):
		images = self.collect_files(dir)
		flow = torch.FloatTensor(channel,len(images),size,size)

		for i in range(len(images)):
			img = Image.open(images[i])
			img = img.convert('RGB')
			img = self.transform(img)
			flow[:,i,:,:] = img

		num_feature = int(len(images)/num_frame)

		res = len(images)%num_frame
		downsampe = []
		for i in range(int(res/2), len(images), int(num_frame/2)):
			downsampe.append(i)

		all_flow = []

		for i in range(0,len(downsampe)-2):
			vid_tensor = flow[:,downsampe[i]:downsampe[i+2],:,:]
			all_flow.append(vid_tensor)

		return all_flow, len(downsampe)-2
