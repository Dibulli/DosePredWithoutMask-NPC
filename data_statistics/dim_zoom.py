import numpy as np
import math
from PIL import Image
import cv2
import os
from scipy.ndimage import zoom

dim_x, dim_y = 1024, \
               1024

def shape_check_and_rescale(voxel, des_dim_y, des_dim_x):
    if np.ndim(voxel) == 3:
        vox_dim_y, vox_dim_x, vox_dim_z = voxel.shape
        if des_dim_y != vox_dim_y or des_dim_x != vox_dim_x:
            vox_data_type = voxel.dtype
            zoom_par = [des_dim_y / vox_dim_y, des_dim_x / vox_dim_x, 1]
            new_voxel = zoom(voxel, zoom_par, output=vox_data_type, order=1)
            return new_voxel
        else:
            return voxel

true_path = "/Users/mr.chai/Desktop/499339.npy"
true_dose = np.load(true_path)
true_dose = true_dose.reshape(512, 512, true_dose.shape[3])
true_reshape = shape_check_and_rescale(true_dose, dim_x, dim_y)
np.save("/Users/mr.chai/Desktop/499339_preddose_1024.npy", true_reshape)