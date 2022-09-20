from lib.dicom.dicom_dose_series import DicomDoseSeries
from lib.dicom.dicom_image_series import DicomCTSeries
from lib.dicom.dicom_directory import DicomDirectory

import numpy as np

"""此代码可以读取单个患者的剂量并按CT坐标插值生成剂量图numpy矩阵，CT和dose可以放在同一个目录下。"""


if __name__ == '__main__':
    url = "/Users/mr.chai/Data/npc_fd/ptv70/20170506/20170626___PINNAPP4___499339"
    out_path = "/Users/mr.chai/Desktop/499339_truedose.npy"
    dicomdir = DicomDirectory(url)
    dicomdir.scan()

    for dicom_dose_series in dicomdir.series_iter(series_type= "RT Dose Storage"):
        # dose_frame_ref_uid = dicom_dose_series.frame_of_reference_uid
        dicom_dose = DicomDoseSeries(dicom_dose_series)
        dicom_dose.load_data()

        for dicom_ct_series in dicomdir.series_iter(series_type= "CT Image Storage"):
            print("debug")

            img_series = DicomCTSeries(dicom_ct_series)
            img_series.load_data()
            dose_img_coord_array = dicom_dose.to_img_voxel(img_series)
            # print(dicom_img.shape)
            # print(dose_img_coord_array)
            print(dose_img_coord_array.shape)
            print(type(dose_img_coord_array))
            print(np.max(dose_img_coord_array))
            np.save(out_path, dose_img_coord_array)




