import matplotlib.pyplot as plt

import numpy as np
import h5py as h5
root = "/Users/mr.chai/Desktop/"
h5_file = "25f3d20b.h5"
h5_path = root + h5_file
h5data = h5.File(h5_path, 'r')
img = h5data["slice_img"]
dose = h5data["slice_dose"]
print(img.shape)
print(np.max(img))
print(dose.shape)
print(np.max(dose))
for z in range(img.shape[2]):
    print(np.max(img[:, :, z]))
