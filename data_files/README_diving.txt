Date: 31/07/17

If you use our dataset, please cite: P. Parmar, B. T. Morris. Learning To Score Olympic Events. CVPR Workshops, 2017, pp. 20-28 
bibtex:
@InProceedings{Parmar_2017_CVPR_Workshops,
author = {Parmar, Paritosh and Tran Morris, Brendan},
title = {Learning to Score Olympic Events},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {July},
year = {2017}
} 

Sport: Diving

No. of Samples: 370 (extended from [1])

Original length, Video sample directory: diving_samples_len_ori
	individual samples: diving_samples_len_ori / xxx.avi
	
Normalized length (151 frames), Video sample directory (used for the LSTM-based frameworks): diving_samples_len_151_lstm
	normalized length individual samples: diving_samples_len_151_lstm / xxx.avi

Scores: diving_overall_scores.mat
Difficulty level: diving_difficulty_level.mat
Execution scores: overall_scores / difficulty_level

Split used in our paper: split_300_70
	training set: split_300_70/training_idx.mat
	testing set: split_300_70/testing_idx.mat
	

References:
[1] H. Pirsiavash, C. Vondrick, and A. Torralba. Assessing the quality of actions. In Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VI, pages 556–571, 2014.