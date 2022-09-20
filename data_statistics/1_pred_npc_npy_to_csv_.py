import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
from scipy.ndimage import zoom

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

x_label = []
raw_path = "/Users/mr.chai/Desktop/mask_npy/499339"

patient_id = raw_path[-6 :]

# true_dose = np.load("/Users/mr.chai/Desktop/499339_truedose.npy")
true_dose = np.load("/Users/mr.chai/Desktop/499339_pred1024.npy")
save_path = '/Users/mr.chai/Desktop/mask_pred_csv/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# flip_dose = np.flip(true_dose, axis=2)
# 确认输入的dose和roi是否能对上。generate的true dose 是需要zflip的
# true dose和pred dose都是正常的，要把roi的npy flip一下就好了。

print(np.max(true_dose))
print(true_dose.shape)
dose_check = shape_check_and_rescale(true_dose, 1024, 1024)

for i in range(90):
    x_label.append(i + 1)


for root, dirs, files in os.walk(raw_path):
    for file in files:
        if file != ".DS_Store":
            print(file[: -8])

            y_label = []
            y_label.append(100)
            dvh_list = []
            dvh_list.append('dose')
            roi = np.load(raw_path + os.sep + file, allow_pickle=True)
            roi = np.flip(roi, axis=2)

            dose_roi = roi * dose_check

            slice_volume = np.count_nonzero(dose_roi)
            max_dose = int(np.ceil(np.max(dose_roi)))

            for i in range(1, max_dose):
                pre_matrix = dose_roi // i
                count_i = np.count_nonzero(pre_matrix)
                count_percentage = count_i / slice_volume * 100
                save_percentage = np.round(count_percentage, 2)
                y_label.append(save_percentage)

            print("[{}] dose 1st period completed!".format(file))

            for j in range(max_dose, 90):
                y_label.append(0)

            print("[{}] dose 2nd period completed!".format(file))

            dvh_list.append(file[: -8])
            print(dvh_list)

            trans_dvh_list = zip(x_label, y_label)
            list_pd = pd.DataFrame(columns=dvh_list, data=trans_dvh_list)
            # print(list_pd)
            list_pd.to_csv(save_path + str(file[: -8]) + ".csv", encoding='gbk')
            print("[{}] save completed!".format(str(file[: -8])))

    print("All file save completed!")


