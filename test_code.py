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

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

from torchvision.transforms import ToPILImage

import csv
import skvideo.io
import cv2

from scipy.signal import resample
def read_split_file(dir):
    res = []
    with open(dir) as f:
        for line in f:
            split = line.split('/')
            res.append(split)
    return res

def get_video_tensor(dir):
	images = collect_files(dir)
	flow = torch.FloatTensor(3,16,160,160)
	seed = np.random.random_integers(0,len(images)-16) #random sampling
	print seed, len(images)
	for i in range(16):
		img = Image.open(images[i+seed])
		img = img.convert('RGB')
		# img = self.transform(img)
		img = transformations(img)
		flow[:,i,:,:] = img

	to_pil_image = ToPILImage()
	img = to_pil_image(flow[:,7,:,:])
	img.show()


transformations = transforms.Compose([transforms.Scale((160,160)),
								transforms.ToTensor()
								])


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

def trans_label(txt):
	# label_list = np.loadtxt(txt)
	label_list = np.genfromtxt(txt, delimiter=' ', dtype=None)
	label_dict = {}
	# num_lab = len(label_list)
	num_lab = 101
	for i in range(num_lab):
		print (label_list[i][1].astype(str), label_list[i][0])
		label_dict[label_list[i][1].astype(str)] = label_list[i][0]
	# print label_dict['HandStandPushups']
	return label_dict

def read_csv_file(dir):
	content = []
	with open(dir) as f:
		f_csv = csv.reader(f)
		headers = next(f_csv)
		a = headers
		for row in f_csv:
			content.append(row)
	return content

def get_vid_dir(index,data_folder,video_content):
	dir = join(data_folder, video_content[index][0])
	dir = join(dir, video_content[index][1]+"*")
	return dir

# def get_video_tensor(video_path):
# 	vidcap = skvideo.io.VideoCapture(video_path)
# 	vidcap.open(video_path)
# 	while True:
# 		flag, bgr_im = vidcap.read()
# 		print flag
# 		if not flag:
# 			break
# 		print bgr_im.shape()

# file = os.path.join('/media/ye/youtube-8/ucfTrainTestlist/testlist01.txt')
# res = read_csv_file(file)
# print res[0][1][:-6]

# file = os.path.join('/media/ye/Seagate Expansion Drive/Kinetics/data/kinetics_train.csv')
# res = read_csv_file(file)
# video_content = res
# data_folder = '/media/ye/youtube-8/kinetics/kinetics_train'
# path = get_vid_dir(0,data_folder,video_content)
# # print path

# path = '/media/ye/youtube-8/kinetics/kinetics_train/abseiling/4YTwq0-73Y_000044_000054.mp4'
# # print path

# get_video_tensor(path)


# dir = os.path.join('/home/ye/Works/C3D-TCN-Keras/frames/v_ApplyEyeMakeup_g01_c02')
# get_video_tensor(dir)

# label_dict = trans_label('./ucfTrainTestlist/classInd.txt')
# print (label_dict)
# label = label_dict['Swing']
# print(label)

a = [i for i in range(0,97)]
a = np.array(a)
print len(a)
l = []
for i in range(0, len(a), len(a)/16):
	l.append(i)
l = l[len(l)-16:]
# a = resample(a, 16)
print l, len(l)