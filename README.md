# Stacking Segment-level P3D for Action Quality Assessment

### Introduction
This repository contains the implementation, models and data (lebel file) for [S3D: Stacking Segment-level P3D for Action Quality Assessment] (https://www.researchgate.net/publication/323074943_S3D_Fusing_Segment-level_P3D_for_Action_Quality_Assessment).

### Requirements:
- PyTorch (with python 3.6)
- Keras
- numpy
- scipy
- sklearn
- skvideo 
- opencv 

### Usage
* The command to train ED-TCN on ResNet features
```
	python seg_train.py 
```
* The command to train P3D-spaced
```
	python train_diving.py --gpuid=0 --stop=0.79
```
* The command to train P3D-center on stage 3
```
	python train_diving.py --tcn_range=3 --downsample=2 --gpuid=0 --stop=0.80
```
* The command to get correlation using SvR/LR (using the extracted P3D features from `./data_files/all_train_v2.npy` and `./data_files/all_test_v2.npy`)
```
	python svr.py
```
### Data
The diving videos are from [UNLV-Dive dataset](http://rtis.oit.unlv.edu/datasets.html). We annotated segmentation labels for this dataset at `./data_files/jump_drop_water_label.txt`

### Models
Models with weights can be downloaded from [google drive](https://drive.google.com/drive/folders/1zC-fghZIKDN5wr4jDLAO_OYAT7Y9ShUo). 
- checkpoint90.tar is trained on stage 1 (jumping)
- checkpoint91.tar is trained on stage 2 (dropping)
- checkpoint79.tar is trained on stage 3 (entering into water)
- checkpoint92.tar is trained on stage 4 (ending)
    
### Acknowledgement
- The P3D model (with weights pre-trained on kinetics) is revised from [P3D-Pytorch](https://github.com/qijiezhao/pseudo-3d-pytorch) by qijiezhao.
- The ED-TCN model is revised from [ED-TCN](https://github.com/colincsl/TemporalConvolutionalNetworks) by colincsl.

### Contact
If there are any questions, please contact me at tytian@outlook.com.
