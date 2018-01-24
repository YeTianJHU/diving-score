import numpy as np
from scipy import io

mat = io.loadmat('diving_overall_scores.mat')

# import h5py
# mat = h5py.File('yourfile.mat')


print(mat.keys())
print(mat.values())

print(mat['overall_scores'].shape)

mat_t = np.transpose(mat['overall_scores'])

np.save('overall_scores.npy', mat_t)

