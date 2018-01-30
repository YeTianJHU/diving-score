from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from i3dpt import Unit3Dpy, I3D

def transfer_model(model,num_classes,model_type):
	if model_type=="P3D":
		num_ftrs=model.fc.in_features
		model.fc=nn.Linear(num_ftrs,num_classes)
	elif model_type=="C3D":
		model.fc8=nn.Linear(4096, num_classes)
	elif model_type=="I3D":
		conv3d_0c_1x1 = Unit3Dpy(
			in_channels=1024,
			out_channels=num_classes,
			kernel_size=(1, 1, 1),
			activation=None,
			use_bias=True,
			use_bn=False)
		model.conv3d_0c_1x1 = conv3d_0c_1x1
	return model

if __name__ == '__main__':

	model = I3D(num_classes=400, modality='rgb')
	model = transfer_model(model, 101, "I3D")
	data=torch.autograd.Variable(torch.rand(10,3,16,224,224))   # if modality=='Flow', please change the 2nd dimension 3==>2
	out=model(data)
	print (out[0].size())
