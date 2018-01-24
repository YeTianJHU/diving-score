#!/usr/bin/env python

import os, sys
import numpy as np
import cv2
# from cv2 import DualTVL1OpticalFlow_create as DualTVL1
import skvideo.io
import time

from os import listdir
from os.path import isfile, join, isdir

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow
##############################################################################################################
def collect_files(dir_name, file_ext=".mp4", sort_files=True):
	allfiles = [os.path.join(dir_name,f) for f in listdir(dir_name) if isfile(join(dir_name,f))]

	these_files = []
	for i in range(0,len(allfiles)):
		_, ext = os.path.splitext(os.path.basename(allfiles[i]))
		if ext == file_ext:
			these_files.append(allfiles[i])

	if sort_files and len(these_files) > 0:
		these_files = sorted(these_files)

	return these_files


##############################################################################################################
def main(args=None, parser=None):

	data_dir = '/home/ye/Works/diving/frames'
	flow_dir = '/home/ye/Works/diving/flow'


	folder = flow_dir
	file_dir = data_dir

	video_files = os.listdir(file_dir)
	# print video_files
	nVideos = len(video_files)

	start_time = time.time()
	for i in range(0,nVideos):
		print i, '/', nVideos

		vid_file = video_files[i]
		bn = os.path.basename(vid_file)
		prefix = os.path.splitext(bn)[0]
		print prefix
		imgae_folder = join(folder, prefix)
		vid_file = join(data_dir, vid_file)
		print vid_file

		if vid_file != '/home/ye/Works/diving/frames/._.DS_Store' and vid_file != '/home/ye/Works/diving/frames/.DS_Store':
			img_list = collect_files(vid_file, file_ext=".jpg", sort_files=True)

			for i in range(len(img_list)-1):
				img_0 = cv2.imread(join(vid_file, img_list[i]))
				img_1 = cv2.imread(join(vid_file, img_list[i+1]))

				frame_0 = cv2.cvtColor(img_0,cv2.COLOR_RGB2GRAY)
				frame_1 = cv2.cvtColor(img_1,cv2.COLOR_RGB2GRAY)

				dtvl1=cv2.createOptFlow_DualTVL1()
				flowDTVL1=dtvl1.calc(frame_0,frame_1,None)

				flow_x=ToImg(flowDTVL1[...,0],20)
				flow_y=ToImg(flowDTVL1[...,1],20)

				if not os.path.exists(imgae_folder):
					os.makedirs(imgae_folder)

				save_x=os.path.join(imgae_folder,'flow_x_{:05d}.jpg'.format(i))
				save_y=os.path.join(imgae_folder,'flow_y_{:05d}.jpg'.format(i))

				cv2.imwrite(save_x, flow_x)
				cv2.imwrite(save_y, flow_y)


	print '\nDONE\n'
	elapsed_time = time.time() - start_time
	print 'time: ', elapsed_time

	return 0


##############################################################################################################
if __name__ == '__main__':
	sys.exit(main())
