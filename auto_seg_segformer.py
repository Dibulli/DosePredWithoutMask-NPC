import logging

import numpy as np
import torch

from lib.color_palette import *
from lib.dicom.dicom_directory import DicomDirectory
from lib.dicom.dicom_image_series import DicomCTSeries
from lib.dicom.dicom_dose_series import DicomDoseSeries
from lib.dicom.dicom_rtst import masks_to_dicomrt
from lib.dicom.dicom_rtst import DicomRTST
from lib.network import segformer_pytorch as segformer

from lib.utilities import shape_check_and_rescale
from lib.utilities import to_abs_path
import os

logger = logging.getLogger(__name__)


def run_auto_seg(cfg, dcm_img_folder=None, output_folder=None):
    if dcm_img_folder is None:
        dcm_img_folder = cfg.predict_dicom_path
    if output_folder is None:
        output_folder = cfg.dose_output_path

    dcm_img_folder = to_abs_path(dcm_img_folder)
    output_folder = to_abs_path(output_folder)
    model_file_path = to_abs_path(cfg.model_file_path)
    print(output_folder)
    # set device before build
    model = segformer.Segformer()

    if cfg.cpu_or_gpu == 'gpu':
        gpu = True
        device = torch.device('cuda:' + str(cfg.gpu_id))
        model.to(device)
    elif cfg.cpu_or_gpu == 'cpu':
        gpu = False
    # model.load_state_dict(torch.load(model_file_path, map_location='cpu'))
    model.load_state_dict(torch.load(model_file_path))

    # model.eval()
    model.train()

    pred_dicomdir = DicomDirectory(dcm_img_folder)
    pred_dicomdir.scan()

    logger.info('start auto segmentation.')
    logger.info('start searching series')
    for dicom_mr_series_base_info in pred_dicomdir.series_iter(series_type='CT Image Storage'):

        dicom_mr_series = DicomCTSeries(dicom_mr_series_base_info)
        dicom_mr_series.load_data()
    # debug
        pt_id = dicom_mr_series.patient_id

        # normalization setting
        if cfg.norm_method == 'fix':
            dicom_mr_series.norm_method = 'fix'
            dicom_mr_series.min_value = cfg.norm_low
            dicom_mr_series.max_value = cfg.norm_high

        slice_img = dicom_mr_series.normalized_voxel_array
        slice_img = shape_check_and_rescale(slice_img, cfg.dim_y, cfg.dim_x)
        half_dim = int((cfg.channels_of_input_images - 1) / 2)
        z_dim = slice_img.shape[2]
        # masks_pred = np.zeros((cfg.channels_of_output_mask, cfg.dim_y, cfg.dim_x, z_dim), dtype=np.float32)
        dose_pred = np.zeros((1, cfg.dim_y, cfg.dim_x, z_dim), dtype=np.float32)

        print("slice_img.shape:[{}]".format(slice_img.shape))

        with torch.no_grad():
            for z in range(z_dim):
            # z = 30
            # inintial input images
            # img = np.zeros((1, cfg.channels_of_input_images, cfg.dim_y, cfg.dim_x), dtype=np.float32)
                img = np.zeros((1, 1, cfg.dim_y, cfg.dim_x), dtype=np.float32)


                # for rel_slice_loc in range(cfg.channels_of_input_images):
                for rel_slice_loc in range(1):

                    z_loc = max(z + rel_slice_loc - half_dim, 0)
                    z_loc = min(z_loc, z_dim - 1)
                    img[0, rel_slice_loc, :, :] = slice_img[:, :, z_loc]
                    logger.info("maximum number of slice [{:}] in imgs is[{:}]".format(z, np.max(slice_img[:, :, z_loc])))
                    logger.info("")
                    # img[rel_slice_loc, :, :] = slice_img[:, :, z_loc]

                # img = torch.from_numpy(img)

                img = torch.from_numpy(img)
                if gpu:
                    img = img.to(device)
                    dose_pred[:, :, :, z] = model(img).cpu().numpy()
                else:
                    dose_pred[:, :, :, z] = model(img).numpy()
                logger.info("maximum dose of slice [{:}] is: [{:}]".format(z,np.max(dose_pred[:, :, :, z])))
                logger.info("")
        logger.info('dose autoseg finished')
        dose_pred.reshape(cfg.dim_y, cfg.dim_x, z_dim)
        np.save(output_folder + os.sep + str(pt_id) + ".npy", dose_pred)
        logger.info('dose save completed')


        # masks_to_dicomrt(dicom_mr_series,
        #                  cfg.roi_dicts,
        #                  masks=masks_pred,
        #                  roi_color_set=set1,
        #                  output_folder=output_folder)

