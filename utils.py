from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
# from i3dpt import Unit3Dpy, I3D

def transfer_model(model,num_classes,model_type):
	if model_type=="P3D":
		num_ftrs=model.fc.in_features
		model.fc=nn.Linear(num_ftrs,num_classes)
	return model

if __name__ == '__main__':

	model = I3D(num_classes=400, modality='rgb')
	model = transfer_model(model, 101, "P3D")
	data=torch.autograd.Variable(torch.rand(10,3,16,160,160))   # if modality=='Flow', please change the 2nd dimension 3==>2
	out=model(data)
	print (out[0].size())
