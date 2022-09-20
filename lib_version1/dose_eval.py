import logging
import torch
import cv2
import os
import numpy as np
from torchsummary import summary
from torch.utils.data import DataLoader
from seg_task.data_statistics.useless_code_temp.seg_data import SegData
from lib.utilities import to_abs_path
from lib.utilities import clean_folder

logger = logging.getLogger(__name__)


def dice_cal_fun(channel, eps=1):
    def sub_dice(input, target):
        iflat = input[:, channel, :, :].reshape(-1, 1)
        tflat = target[:, channel, :, :].reshape(-1, 1)
        intersection = (iflat * tflat).sum()
        return (2. * intersection + eps) / (iflat.sum() + tflat.sum() + eps)
    return sub_dice

class dose_eval():
    def __init__(self, cfg):
        self.cfg = cfg
        self.channel_info = {'ptvap704': 0, 'ptvap660': 1, 'ptvap608': 2, 'ptvap600': 3,
                             'ptvap560': 4, 'ptvap540': 5}
        self.test_ds = SegData(self.cfg, 'test')
        self.test_loader = DataLoader(self.test_ds, batch_size=self.cfg.batch_size,
                                      num_workers=self.cfg.train_loader_works,
                                      pin_memory=True)
        self.model_file_path = to_abs_path(cfg.model_file_path)
        if cfg.cpu_or_gpu == 'gpu':
            self.gpu = True
            self.device = torch.device('cuda:' + str(cfg.gpu_id))
        else:
            self.gpu = False
            self.device = 'cpu'

        self.model = None
        self._model_set()


    def _model_build(self):
        if 'model_type' in self.cfg:
            if self.cfg.model_type == 'unet_res':
                self.model_type = 'unet_res'
                from lib.network.unet_res import UNet
            elif self.cfg.model_type == 'unet_res_big':
                self.model_type = 'unet_res_big'
                from lib.network.unet_res_big import UNet
            elif self.cfg.model_type == 'unet_mask':
                self.model_type = 'unet_mask'
                from lib.network.unet_mask import UNet
            elif self.cfg.model_type == 'unet_attention':
                self.model_type = 'unet_attention'
                from lib.network.unet_attention import UNet
            elif self.cfg.model_type == 'unet_type2':
                self.model_type = 'unet_type2'
                from lib.network.unet_type2 import UNet
            elif self.cfg.model_type == 'unet_deeper':
                self.model_type = 'unet_deeper'
                from lib.network.unet_deeper import UNet
            elif self.cfg.model_type == 'wnet':
                self.model_type = 'wnet'
                from lib.network.wnet import Wnet as UNet
            else:
                self.model_type = 'unet'
                from lib.network.unet import UNet
        else:
            self.model_type = 'unet'
            from lib.network.unet import UNet

        self.model = UNet(self.cfg.channels_of_input_images, self.cfg.channels_of_output_mask, 1)
        summary(self.model.to(self.device), (6, 512, 512))

        # set device before build
        if self.gpu:
            self.model.to(self.device)

    def _load_weight(self):
        logger.info('load weight')
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))
        logger.info('load weight successful')

    def _model_set(self):
        self._model_build()
        self._load_weight()

    def dose_show(self):
        # set model to evaluation
        self.model.eval()
        check_img_path = to_abs_path(self.cfg.test_out_path)

        set1 = [[255, 127, 53],
                [84, 138, 255],
                [99, 169, 0],
                [178, 67, 201],
                [36, 136, 0],
                [219, 26, 156],
                [0, 200, 131],
                [244, 25, 126],
                [61, 221, 185],
                [247, 59, 71],
                [1, 211, 232],
                [207, 76, 0],
                [1, 130, 210],
                [255, 184, 33],
                [178, 140, 255],
                [222, 199, 55],
                [24, 81, 157],
                [143, 149, 0],
                [202, 179, 255],
                [88, 126, 0],
                [255, 127, 185],
                [0, 138, 97],
                [255, 108, 83],
                [1, 151, 212],
                [160, 32, 18],
                [86, 122, 77],
                [255, 126, 108],
                [205, 202, 122],
                [137, 52, 91],
                [255, 180, 113],
                [154, 38, 57],
                [104, 79, 22]]

        clean_folder(check_img_path)

        # loop batch
        with torch.no_grad():
            for batch_i, batch_data in enumerate(self.test_loader):

                batch_img = batch_data[0].numpy()
                batch_mask = batch_data[1].numpy()
                batch_pt_id = batch_data[3]
                batch_z = batch_data[4].numpy()
                batch_dose = batch_data[2].numpy()
                if self.gpu:
                    batch_img_gpu = batch_data[0].to(self.device)
                pred = self.model(batch_img_gpu)
                dose_pred = pred[1].cpu().numpy()
                mask_pred = pred[0].cpu().numpy()

                for sample_i in range(len(batch_pt_id)):
                    logger.info('batch [{}] sample [{}]'.format(
                        str(batch_i), str(sample_i)))
                    pt_id = batch_pt_id[sample_i]

                    combine_check_img_rgb = None
                    combine_pred_img_rgb = None
                    # check_img = np.array(batch_img[sample_i, :, :, :].transpose(1, 2, 0) * 255,
                    #                      dtype=np.uint8)
                    check_img = np.array(batch_img[sample_i, round(self.cfg.channels_of_input_images / 2), :, :] * 255,
                                         dtype=np.uint8)
                    check_img = check_img.reshape((check_img.shape[0], check_img.shape[1], 1))
                    check_img_rgb = cv2.cvtColor(check_img, cv2.COLOR_GRAY2RGB)
                    check_dose = np.array(batch_dose[sample_i, :, :, :].transpose(1, 2, 0) * 255,
                                          dtype=np.uint8)
                    check_pred_dose = np.array(dose_pred[sample_i, :, :, :].transpose(1, 2, 0) * 255,
                                          dtype=np.uint8)
                    check_dose_rgb = cv2.applyColorMap(check_dose, cv2.COLORMAP_JET)
                    check_pred_dose_rgb = cv2.applyColorMap(check_pred_dose, cv2.COLORMAP_JET)

                    for mask_channel in range(5):
                        roi_color = set1[mask_channel]
                        roi_check_img = batch_mask[sample_i, mask_channel, :, :]

                        roi_check_img_rgb = np.repeat(roi_check_img[:, :, np.newaxis], 3, axis=2)
                        roi_check_img_rgb = roi_check_img_rgb * np.array(roi_color, dtype=int)
                        roi_check_img_rgb = np.array(roi_check_img_rgb, dtype=np.uint8)

                        if combine_check_img_rgb is None:
                            combine_check_img_rgb = cv2.addWeighted(check_img_rgb, 1.0, roi_check_img_rgb,
                                                                    0.7, 0)
                        else:
                            combine_check_img_rgb = cv2.addWeighted(combine_check_img_rgb, 1.0, roi_check_img_rgb,
                                                                    0.7, 0)
                        pred_roi_color = set1[mask_channel]
                        pred_roi_check_img = (mask_pred[sample_i, mask_channel, :, :] > 0.5)
                        dice = np.zeros((len(batch_pt_id)))
                        dice[sample_i] = dice_cal_fun(channel=mask_channel)(batch_data[1][sample_i: sample_i + 1].cpu(),
                                                                             pred[0][sample_i: sample_i + 1].cpu()).item()

                        pred_roi_check_img_rgb = np.repeat(pred_roi_check_img[:, :, np.newaxis], 3, axis=2)
                        pred_roi_check_img_rgb = pred_roi_check_img_rgb * np.array(pred_roi_color, dtype=int)
                        pred_roi_check_img_rgb = np.array(pred_roi_check_img_rgb, dtype=np.uint8)

                        if combine_pred_img_rgb is None:
                            combine_pred_img_rgb = cv2.addWeighted(check_img_rgb, 1.0, pred_roi_check_img_rgb,
                                                                    0.7, 0)
                        else:
                            combine_pred_img_rgb = cv2.addWeighted(combine_pred_img_rgb, 1.0, pred_roi_check_img_rgb,
                                                                    0.7, 0)

                    combine_out = np.zeros((4 * combine_check_img_rgb.shape[0], combine_check_img_rgb.shape[1], 3))
                    combine_out[0: combine_check_img_rgb.shape[0], :, :] = combine_check_img_rgb
                    combine_out[combine_check_img_rgb.shape[0]: 2 * combine_check_img_rgb.shape[0], :, :] = combine_pred_img_rgb
                    combine_out[2 * combine_check_img_rgb.shape[0]: 3 * combine_check_img_rgb.shape[0], :, :] = check_dose_rgb
                    combine_out[3 * combine_check_img_rgb.shape[0]:, :, :] = check_pred_dose_rgb
                    dice_i = str(dice[sample_i])
                    file_name = '[{0:s}]_b[{1:s}]_s[{2:s}]_dice[{3:s}]img.png'.format(
                        pt_id, str(batch_i), str(sample_i), dice_i)
                    file_path = check_img_path + os.sep + file_name
                    cv2.imwrite(file_path, combine_out)
                    # dose_file_name = '[{0:s}]_b[{1:s}]_s[{2:s}]_dose.png'.format(
                    #     pt_id, str(batch_i), str(sample_i))
                    # dose_file_path = check_img_path + os.sep + dose_file_name
                    # cv2.imwrite(dose_file_path, check_dose_rgb)
                    # dose_file_name = '[{0:s}]_b[{1:s}]_s[{2:s}]_dose_pred.png'.format(
                    #     pt_id, str(batch_i), str(sample_i))
                    # dose_file_path = check_img_path + os.sep + dose_file_name
                    # cv2.imwrite(dose_file_path, check_pred_dose_rgb)
                    # def cal_dvh(self):
                    #     # set model to evaluation
                    #     self.model.eval()
                    #
                    #     # initial training loss
                    #     total_sample_size = 0
                    #
                    #     # loop batch
                    #     with torch.no_grad():
                    #         for batch_i, batch_data in enumerate(self.test_loader):
                    #
                    #             imgs = batch_data[0]
                    #             mask = batch_data[1]
                    #             input_data = np.concatenate((imgs, mask), axis=1)
                    #             input_data = torch.from_numpy(input_data)
                    #             dose = batch_data[2]
                    #
                    #             sample_size = imgs.size(0)
                    #             total_sample_size = total_sample_size + sample_size
                    #
                    #             if self.gpu:
                    #                 input_data = input_data.to(self.device)
                    #                 dose = dose.to(self.device)
                    #
                    #             dose_pred = self.model(input_data)[1]


    def roi_show(self):
        # set model to evaluation
        self.model.eval()
        check_img_path = to_abs_path(self.cfg.test_out_path)

        set1 = [[255, 127, 53],
                [84, 138, 255],
                [99, 169, 0],
                [178, 67, 201],
                [36, 136, 0],
                [219, 26, 156],
                [0, 200, 131],
                [244, 25, 126],
                [61, 221, 185],
                [247, 59, 71],
                [1, 211, 232],
                [207, 76, 0],
                [1, 130, 210],
                [255, 184, 33],
                [178, 140, 255],
                [222, 199, 55],
                [24, 81, 157],
                [143, 149, 0],
                [202, 179, 255],
                [88, 126, 0],
                [255, 127, 185],
                [0, 138, 97],
                [255, 108, 83],
                [1, 151, 212],
                [160, 32, 18],
                [86, 122, 77],
                [255, 126, 108],
                [205, 202, 122],
                [137, 52, 91],
                [255, 180, 113],
                [154, 38, 57],
                [104, 79, 22]]

        clean_folder(check_img_path)

        # loop batch
        with torch.no_grad():
            for batch_i, batch_data in enumerate(self.test_loader):

                batch_img = batch_data[0].numpy()
                batch_mask = batch_data[1].numpy()
                batch_pt_id = batch_data[2]
                batch_z = batch_data[3].numpy()
                if self.gpu:
                    batch_img_gpu = batch_data[0].to(self.device)
                pred = self.model(batch_img_gpu)
                mask_pred = pred.cpu().numpy()

                for sample_i in range(len(batch_pt_id)):
                    logger.info('batch [{}] sample [{}]'.format(
                        str(batch_i), str(sample_i)))
                    pt_id = batch_pt_id[sample_i]

                    combine_check_img_rgb = None
                    combine_pred_img_rgb = None
                    check_img = np.array(batch_img[sample_i, round(self.cfg.channels_of_input_images / 2), :, :] * 255,
                                         dtype=np.uint8)
                    check_img_rgb = cv2.cvtColor(check_img, cv2.COLOR_GRAY2RGB)


                    for mask_channel in range(5):
                        roi_color = set1[mask_channel]
                        roi_check_img = batch_mask[sample_i, mask_channel, :, :]

                        roi_check_img_rgb = np.repeat(roi_check_img[:, :, np.newaxis], 3, axis=2)
                        roi_check_img_rgb = roi_check_img_rgb * np.array(roi_color, dtype=int)
                        roi_check_img_rgb = np.array(roi_check_img_rgb, dtype=np.uint8)

                        if combine_check_img_rgb is None:
                            combine_check_img_rgb = cv2.addWeighted(check_img_rgb, 1.0,
                                                                    roi_check_img_rgb,
                                                                    0.7, 0)
                        else:
                            combine_check_img_rgb = cv2.addWeighted(combine_check_img_rgb, 1.0,
                                                                    roi_check_img_rgb,
                                                                    0.7, 0)
                        pred_roi_color = set1[mask_channel]
                        pred_roi_check_img = (mask_pred[sample_i, mask_channel, :, :] > 0.5)
                        dice = np.zeros((len(batch_pt_id)))
                        dice[sample_i] = dice_cal_fun(channel=mask_channel)(
                            batch_data[1][sample_i: sample_i + 1].cpu(),
                            pred[sample_i: sample_i + 1].cpu()).item()

                        pred_roi_check_img_rgb = np.repeat(pred_roi_check_img[:, :, np.newaxis], 3,
                                                           axis=2)
                        pred_roi_check_img_rgb = pred_roi_check_img_rgb * np.array(pred_roi_color,
                                                                                   dtype=int)
                        pred_roi_check_img_rgb = np.array(pred_roi_check_img_rgb, dtype=np.uint8)

                        if combine_pred_img_rgb is None:
                            combine_pred_img_rgb = cv2.addWeighted(check_img_rgb, 1.0,
                                                                   pred_roi_check_img_rgb,
                                                                   0.7, 0)
                        else:
                            combine_pred_img_rgb = cv2.addWeighted(combine_pred_img_rgb, 1.0,
                                                                   pred_roi_check_img_rgb,
                                                                   0.7, 0)

                    combine_out = np.zeros(
                        (2 * combine_check_img_rgb.shape[0], combine_check_img_rgb.shape[1], 3))
                    combine_out[0: combine_check_img_rgb.shape[0], :, :] = combine_check_img_rgb
                    combine_out[combine_check_img_rgb.shape[0]: 2 * combine_check_img_rgb.shape[0], :,
                    :] = combine_pred_img_rgb
                    dice_i = str(dice[sample_i])
                    file_name = '[{0:s}]_b[{1:s}]_s[{2:s}]_dice[{3:s}]img.png'.format(
                        pt_id, str(batch_i), str(sample_i), dice_i)
                    file_path = check_img_path + os.sep + file_name
                    cv2.imwrite(file_path, combine_out)







