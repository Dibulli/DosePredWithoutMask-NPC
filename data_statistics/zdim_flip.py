import numpy as np
import math
from PIL import Image
import cv2
import os
from scipy.ndimage import zoom


true_path = "/Users/mr.chai/Desktop/499339_truedose_512.npy"
true_dose = np.load(true_path)

flip_dose = np.flip(true_dose, axis=2)
np.save("/Users/mr.chai/Desktop/499339_truedose_512_flip.npy", flip_dose)