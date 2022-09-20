import pydicom
import os
import shutil

file_path = "/Users/mr.chai/Data/NPC_by_szh"
count = 0

for root, dirs, files in os.walk(file_path):
    if "CT" in str(root):
        print(root)
        print("dirs:", dirs)
        for file in files:
            if "00095.DCM" in file:
                print("file:", file)
                print("dirs:", dirs)
                print("root:", root)
                dcm_ct_path = root + os.sep + file
                dcm_dose_dir = root[: -2] + "RTdose"
                dose_file = os.listdir(dcm_dose_dir)[0]
                print(dose_file)
                dcm_dose_path = dcm_dose_dir + os.sep + dose_file

                ct_ds = pydicom.dcmread(dcm_ct_path, force=True)
                slice_thickness = ct_ds.SliceThickness

                dose_ds = pydicom.dcmread(dcm_dose_path, force=True)
                dose_ds.SliceThickness = slice_thickness

                pydicom.dcmwrite(dcm_dose_path, dose_ds)
                count += 1
                print(count)


