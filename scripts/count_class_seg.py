from PIL import Image
import cv2
import numpy as np
import scipy
import os
from glob import glob

env = ["Kite_training", "PLE_training"]
seasons1 = ["cloudy", "foggy", "sunny", "sunset"]
seasons2 = ["fall", "spring", "winter"]

count = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for seas in seasons1:
    print("Processing: ", seas)
    db_path = os.path.join(*["/media/DATA_4TB/Yara/midair",env[0]])
    dirs_path = os.path.join(*[db_path, seas, "segmentation"])+"/*/"
    print("Dir Path: ", dirs_path)
    dirs = glob(os.path.join(*[db_path, seas, "segmentation"])+"/*/")
    print("Directories: ", dirs)
    for d in dirs:
        #imgs = os.listdir(os.path.join(*[db_path,env[0], seas, "segmentation", d]))
        imgs = os.listdir(d)
        for im in imgs:
            inp_path = os.path.join(*[d, im])
            im = cv2.imread(inp_path)
            #print("SIZE = ",np.shape(im))
            #im1 = cv2.resize(im, (528,384), interpolation = cv2.INTER_NEAREST)
            #print("SIZE = ",np.shape(im1))
            vals, counts = np.unique(im[:,:,0], return_counts = True)
            for i, v in enumerate(vals):
                count[v] = count[v] + counts[i]
            
for seas in seasons2:
    print("Processing: ", seas)
    db_path = os.path.join(*["/media/DATA_4TB/Yara/midair",env[1]])
    dirs = glob(os.path.join(*[db_path, seas, "segmentation"])+"/*/")
    print("Directories: ", dirs)
    for d in dirs:
        #imgs = os.listdir(os.path.join(*[db_path, env[1], seas, "segmentation", d]))
        imgs = os.listdir(d)
        for im in imgs:
            inp_path = os.path.join(*[ d, im])
            im = cv2.imread(inp_path)
            #print("SIZE = ",np.shape(im))
            #im1 = cv2.resize(im, (528,384), interpolation = cv2.INTER_NEAREST)
            #print("SIZE = ",np.shape(im1))
            vals, counts = np.unique(im[:,:,0], return_counts = True)
            for i, v in enumerate(vals):
                count[v] = count[v] + counts[i]
print("TOTAL")
print("Count0 = ",count[0])
print("Count1 = ",count[1])
print("Count2 = ",count[2])
print("Count3 = ",count[3])
print("Count4 = ",count[4])
print("Count5 = ",count[5])
print("Count6 = ",count[6])
print("Count7 = ",count[7])
print("Count8 = ",count[8])
