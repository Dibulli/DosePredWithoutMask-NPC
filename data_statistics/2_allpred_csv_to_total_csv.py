import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

raw_path = "/Users/mr.chai/Desktop/mask_pred_csv"
count = len(os.listdir(raw_path))
print(count)
mask = [0] * count

for root, dirs, files in os.walk(raw_path):
    files = [f for f in files if not f[0] == '.']
    for file in files:
        mask[0] = pd.read_csv(raw_path + os.sep + file, usecols=[1])
        break
    for i, file in enumerate(files):
        mask[i + 1] = pd.read_csv(raw_path + os.sep + file, usecols=[2])

    all_csv = pd.concat(mask, axis=1)
    print(all_csv)
    all_csv.to_csv(raw_path + os.sep + "total_pred.csv")



