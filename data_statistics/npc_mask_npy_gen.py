import pydicom
from lib.dicom_helper import DicomDirectory, DicomImage
from lib.dicom_helper import DicomRTst
from lib.dicom_helper import rescale_resolution
from lib.dicom_helper import DicomDose
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import shutil
from scipy.ndimage import zoom
import logging
import csv

# logger = logging.getLogger(__name__)


def zflip_data(data):
    print(data.shape)
    zflip_data = np.zeros_like(data)
    for z in range(data.shape[0]):
        zflip_data[z, :, :] = data[data.shape[0] - z - 1, :, :]
    zflip_data = zflip_data.transpose((1, 2, 0))
    print(zflip_data.shape)
    print(np.max(zflip_data))
    zflip_data *= 10000
    print(np.max(zflip_data))

    return zflip_data

url = '/Users/mr.chai/Data/npc_fd/ptv70/20170506/20170626___PINNAPP4___499339'
save_directory = "/Users/mr.chai/Desktop/mask_npy/"

dicomdir = DicomDirectory(url)
all_rtst_file = dicomdir.get_all_rtst_uid()
print("start scan")
ptv704_count = 0


for j in range(len(all_rtst_file)):
    rtst_file = all_rtst_file[j]  # patient number
    dicom_rtst = DicomRTst(rtst_file)
    pt_id = dicom_rtst.patient_id
    rtst_ref_series = dicom_rtst.ref_series
    file_list = dicomdir.get_all_img_files_name_by_img_series_uid(rtst_ref_series)
    dicom_img = DicomImage(file_list)
    dicom_img.create_3D_image()
    dicom_rtst.read_ref_series(file_list)
    dicom_img.image_norm()
    img = dicom_img.image_3d_norm

    roi_names = dicom_rtst.roi_list
    print(roi_names)

    npc_roi_custom = ['body',
                      'r-lens', 'l-lens', 'chaiasm','l-nerve',
                      'r-nerve', 'oralcavity', 'larynx', 'parotid-r', 'parotid-l',
                      'sc', 'brainstem',
                      'ptvap704', 'ptvap600', 'ptvap540']

    os.mkdir(save_directory + pt_id + os.sep)
    for roi_name in npc_roi_custom:
        dicom_rtst.create_3d_mask(roi_name, True)
        roi = dicom_rtst.roimask[roi_name]
        save_path = save_directory + pt_id + os.sep + roi_name + '.npy'
        np.save(save_path, roi)
        print("{} saved!".format(roi_name))



    print("save completed!")











