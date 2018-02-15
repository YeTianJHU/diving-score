import numpy as np

# file_name = './jump_drop_water_new.txt'
# save_name = './time_span.npy'
# time = np.loadtxt(file_name, delimiter=',',dtype='int')


def to_vector(mat):
	"""Convert categorical data into vector.
	Args:
		mat: onr-hot categorical data.
	Returns:
		out2: vectorized data."""
	out = np.zeros((mat.shape[0],mat.shape[1]))
	out2 = np.zeros((mat.shape[0]))
	for i in range(mat.shape[0]):
		for n, j in enumerate(mat[i]):
			if np.any(j == (np.amax(mat[i]))):
				out[i][n] = 1
				out2[i] = n

	return out2

# all_range = []
# for i in range(len(time)):
# 	time_range = []
# 	for j in range(time[i][0]):
# 		time_range.append(0)
# 	for j in range(time[i][0], time[i][1]):
# 		time_range.append(1)
# 	for j in range(time[i][1], time[i][2]):
# 		time_range.append(2)
# 	for j in range(time[i][2], time[i][3]):
# 		time_range.append(3)
# 	for j in range(time[i][3],160):
# 		time_range.append(4)
# 	print (i, len(time_range))
# 	all_range.append(time_range)

# all_range = np.array(all_range)
# print (all_range.shape)
# print (all_range[3])

# np.save(save_name, all_range)

# data = np.load('tcn_output.npy')

# vec_data = []
# for i in data:
# 	vec_data.append(to_vector(i))

# np.save('./data_files/tcn_time_range.npy', vec_data)

vec_data = np.load('./data_files/tcn_time_range.npy')

note_all = []
for idx, video in enumerate(vec_data):
	note = []
	for i in range(len(video)-1):
		if video[i] != video[i+1]:
			note.append(i)
	print (idx+1, note)
	note_all.append(note)
print (len(note_all))
np.save('./data_files/tcn_time_point.npy', note_all)


