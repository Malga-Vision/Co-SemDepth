from PIL import Image
import cv2
import numpy as np
import scipy
import os
from sklearn import metrics
import matplotlib.pyplot as plt


def map_midair(a):
    #print(a)
    if (a == [127,175,230]).all():
        return 0
    elif (a == [75,163,185]).all():
        return 1
    elif (a == [50,128,0]).all():
        return 2
    elif (a == [53,94,59]).all():
        return 3
    elif (a == [237,125,49]).all():
        return 4
    elif (a == [70,70,70]).all():
        return 5
    elif (a == [128,64,128]).all():
        return 6
    elif (a == [64,64,128]).all():
        return 7
    elif (a == [128,64,64]).all():
        return 8



inp_path1 = "/media/DATA_4TB/Yara/results/gt_sem"
inp_path2 = "/media/DATA_4TB/Yara/results/seg_estimation"
imgs = os.listdir(inp_path1)
#imgs = os.listdir(inp_path2)
actual = np.array([])
predicted = np.array([])

idx = 0
for im in imgs:
    if idx % 100 == 0:
        print(im)
        im1_path = os.path.join(*[inp_path1, im])
        im1 = cv2.imread(im1_path)
        im2_path = os.path.join(*[inp_path2, im])
        im2 = cv2.imread(im2_path)
        im_upd1 = np.apply_along_axis(map_midair, -1,im1)
        im_upd2 = np.apply_along_axis(map_midair, -1,im2)
        actual = np.append(actual,im_upd1.flatten())
        predicted = np.append(predicted,im_upd2.flatten())
        
    idx = idx+1
    
    

confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display= metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
#cm_display= metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Sky","Water","Trees","Dirt Ground","Vegetation","Rocks","Road","Others"])
#cm_display= metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Sky","Water","Trees","Land","Vehicles","Rocks","Road","Construction","Others"])

cm_display.plot()
plt.show()
