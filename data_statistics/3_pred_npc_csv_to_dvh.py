import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd

# name037770 = ['dose', 'PTV-C-60', 'PTV-G-70.4', 'PTV-LN-66', 'SC', 'STEM', 'L-lobe', 'R-lobe',
#         'L-Len', 'R-Len', 'L-nerve', 'R-nerve', 'L-pariod', 'R-paroid',
#         'Oral', 'Larynx', 'PTV-LN-54', 'body']

name1 = ['body',
        'r-lens', 'l-lens', 'chaiasm','l-nerve',
        'r-nerve', 'oralcavity', 'larynx', 'parotid-r', 'parotid-l',
        'sc', 'brainstem',
        'ptvap704', 'ptvap600', 'ptvap540']
# name1 = ['L-pariod','PTV-C-60','PTV-G-70.4',  'PTV-LN-54','PTV-LN-66','R-paroid', 'SC', 'STEM']
# name1 = ['PTV-LN-54', 'R-paroid', 'L-pariod', 'STEM','SC', 'PTV-LN-66','PTV-C-60', 'PTV-G-70.4']
# name2 = ['L-lobe', 'R-lobe', 'L-Len', 'R-Len', 'L-nerve', 'R-nerve', 'Oral', 'Larynx']

colorset1 = ['blue', 'red', 'green', 'orange', 'darkblue', 'darkgreen', 'darkred', 'cyan',
             'lightyellow', 'pink', 'gray', "blueviolet", 'purple', 'lavender', 'crimson']
# colorset2 = ['blue', 'red', 'green', 'orange', 'darkblue', 'darkgreen', 'darkred', 'cyan']
dic1 = dict(zip(name1, colorset1))
# dic2 = dict(zip(name2, colorset2))

true_dose = pd.read_csv("/Users/mr.chai/Desktop/total_true.csv")
pre_dose = pd.read_csv("/Users/mr.chai/Desktop/total_pred.csv")
x = pre_dose["dose"]

for roi in name1:
    tru = true_dose[roi]
    pre = pre_dose[roi]
    color1 = dic1[roi]
    plt.plot(x, tru, c=color1, linestyle='-', marker='^', linewidth=1, markersize=3, label=roi + "-manual")
    plt.plot(x, pre, c=color1, linestyle='--', linewidth=1, markersize=3, label=roi + "-predict")

plt.axis([0, 90, 0, 102])


plt.xlabel("dose(Gy)")
plt.ylabel("volume(%)")
plt.legend(loc="best")
plt.title("DVH")

# plt.savefig("/Users/mr.chai/Desktop/dvh.jpg", dpi = 1000, bbox_inches = 'tight')
plt.show()
# r1024 = np.load("/Users/mr.chai/Desktop/ptv70_1024.npy")
# true_dose = np.load("/Users/mr.chai/Desktop/NPC_clinical_dose/RT037622_ori_dim512.npy")
# b512 = np.load("/Users/mr.chai/Desktop/body.npy")
# b1024 = np.load("/Users/mr.chai/Desktop/body_1024.npy")


