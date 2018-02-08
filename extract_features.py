import argparse
import logging
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

from p3d_model import P3D199, C3D, get_optim_policies
from utils import transfer_model
from dataset import divingDataset



data_file = './frames'
feature_folder = './tcn_p3d_features'
range_file = './tcn_time_point.npy'
channel = 3
size = 160
model_input_length = 16
model_type = 'P3D'
use_cuda = 1
load = 31
tcn_range = 3

def collect_files(dir_name, file_ext=".jpg", sort_files=True):
	allfiles = [os.path.join(dir_name,f) for f in listdir(dir_name) if isfile(join(dir_name,f))]

	these_files = []
	for i in range(0,len(allfiles)):
		_, ext = os.path.splitext(os.path.basename(allfiles[i]))
		if ext == file_ext:
			these_files.append(allfiles[i])

	if sort_files and len(these_files) > 0:
		these_files = sorted(these_files)

	return these_files

transformations = transforms.Compose([transforms.Scale((size,size)),
								transforms.ToTensor(),
								transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
								])

def get_range_tensor(dir, num_frame, channel, size, start, transformations):
	images = collect_files(dir)

	if start < 0:
		start = 0
	if start + 15 > len(images):
		start = int(len(images) - 16)

	flow = torch.FloatTensor(channel,num_frame,size,size)

	for i in range(num_frame):
		img = Image.open(images[i+start])
		img = img.convert('RGB')
		img = transformations(img)
		flow[:,i,:,:] = img

	return flow


# Initial the model
if model_type=="P3D":
	model = P3D199(pretrained=False,num_classes=400)
elif model_type=="C3D":
	model = C3D()
elif model_type=="I3D":
	model = I3D(num_classes=101, modality='rgb')
else:
	logging.error("No such model: {0}".format(model_type))


model = transfer_model(model,num_classes=1, model_type=model_type)

if use_cuda > 0:
	model.cuda()

logging.info("=> loading checkpoint"+str(load)+".tar")
checkpoint = torch.load('checkpoint'+str(load)+'.tar')
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
model.eval()

if not isdir(feature_folder):
	os.mkdir(feature_folder)

videos = sorted(listdir(data_file))
print videos
time_range = np.load(range_file)


n_vid = 0
for idx, video in enumerate(videos):
	video_path = join(data_file, video)
	vid_range = time_range[idx]


	feature_vid_path = join(feature_folder, video)

	images = collect_files(video_path)

	if len(vid_range) != 4:
		vid_range = np.insert(vid_range, 0,0)
	if tcn_range != 4:
		mid = int((vid_range[tcn_range-1]+vid_range[tcn_range])/2)
		start = int(mid-8)
	else: 
		start = int(vid_range[4])

	vid_tensor = get_range_tensor(dir=video_path, num_frame=16, channel=3, size=160, start=start, transformations=transformations)

	vid_tensor.unsqueeze_(0)

	if use_cuda:
		vid_tensor = Variable(vid_tensor).cuda()
	else:
		vid_tensor = Variable(vid_tensor)

	test_output = model(vid_tensor)
	feature = test_output[1]
	np_feature =  feature.data.cpu().numpy()[0]
	score = test_output[0]
	np_score = score.data.cpu().numpy()[0]
	features = [np_feature, np_score]

	features = np.array(features)

	np.save(feature_vid_path+'_'+str(tcn_range), features)
	n_vid += 1
	print (video),('|'),(n_vid),('done')


#######################################################################33
# if not isdir(feature_folder):
# 	os.mkdir(feature_folder)

# videos = listdir(data_file)
# n_vid = 0
# for video in videos:
# 	video_path = join(data_file, video)

# 	feature_vid_path = join(feature_folder, video)

# 	images = collect_files(video_path)

# 	flow = torch.FloatTensor(channel,len(images),size,size)

# 	for i in range(len(images)):
# 		img = Image.open(images[i])
# 		img = img.convert('RGB')
# 		img = transformations(img)
# 		flow[:,i,:,:] = img

# 	num_feature = int(len(images)/model_input_length)

# 	res = len(images)%model_input_length
# 	downsampe = []
# 	for i in range(int(res/2), len(images), int(model_input_length/2)):
# 		downsampe.append(i)

# 	features = []
# 	for i in range(0,len(downsampe)-2,2):
# 		vid_tensor = flow[:,downsampe[i]:downsampe[i+2],:,:]
# 		vid_tensor.unsqueeze_(0)

# 		if use_cuda:
# 			print ('use cuda')
# 			vid_tensor = Variable(vid_tensor).cuda()
# 		else:
# 			vid_tensor = Variable(vid_tensor)

# 		test_output = model(vid_tensor)
# 		feature = test_output[1]
# 		np_feature =  feature.data.cpu().numpy()[0]
# 		score = test_output[0]
# 		np_score = score.data.cpu().numpy()[0]
# 		features.append([np_feature, np_score])

# 	features = np.array(features)

# 	# np.save(feature_vid_path, features)
# 	n_vid += 1
# 	print (video),('|'),(n_vid),('done')

