import numpy as np
import cv2
import os
import tensorflow as tf

db_path = "/media/DATA_4TB/Yara/cityscapes/val/segmentation"
envs = os.listdir(db_path)

class_index = {
        0: [19 ,(0, 0, 0),  'void'],
        1: [19, (0, 0, 0),  'void'],
        2: [19, (0, 0, 0),  'void'],
        3: [19, (0, 0, 0),  'void'],
        4: [19, (0, 0, 0),  'void'],
        5: [19, (0, 0, 0),  'void'],
        6: [19, (0, 0, 0),  'void'],
        7: [0, (128, 64, 128),  'road'],
        8: [1, (244, 35, 232),  'sidewalk'],
        9: [19, (0, 0, 0),  'void'],
        10: [19, (0, 0, 0),  'void'],
        11: [2, (70, 70, 70),  'building'],
        12: [3, (102, 102, 156),  'wall'],
        13: [4, (190, 153, 153),  'fence'],
        14: [19, (0, 0, 0),  'void'],
        15: [19, (0, 0, 0),  'void'],
        16: [19, (0, 0, 0),  'void'],
        17: [5, (153, 153, 153),  'pole'],
        18: [19, (0, 0, 0),  'void'],
        19: [6, (250, 170, 30),  'traffic light'],
        20: [7, (220, 220, 0),  'traffic sign'],
        21: [8, (107, 142, 35),  'vegetation'],
        22: [9, (152, 251, 152),  'terrain'],
        23: [10, (70, 130, 180),  'sky'],
        24: [11, (220, 20, 60),  'person'],
        25: [12, (255, 0, 0),  'rider'],
        26: [13, (0, 0, 142),  'car'],
        27: [14, (0, 0, 70),  'truck'],
        28: [15, (0, 60, 100),  'bus'],
        29: [19, (0, 0, 0),  'void'],
        30: [19, (0, 0, 0),  'void'],
        31: [16, (0, 80, 100),  'train'],
        32: [17, (0, 0, 230),  'motorcycle'],
        33: [18, (119, 11, 32),  'bicycle'],
        -1: [19, (0, 0, 0),  'void']
        }
for nset, env in enumerate(envs):
    env_path = db_path+"/"+env
    imgs = os.listdir(env_path)
    
    for im in imgs:
        if "gtFine_labelIds.png" in im:
            im_path = env_path + "/" + im
            file = tf.io.read_file(im_path)
            image = tf.io.decode_png(file)
            semantic_arr = np.array(image)
            
            for i in range(np.shape(semantic_arr)[0]):
                for j in range(np.shape(semantic_arr)[1]):
                    semantic_arr[i][j][0] = class_index[semantic_arr[i][j][0]][0]
            
            
            sem1 = tf.convert_to_tensor(semantic_arr)
            
            tf.keras.utils.save_img(im_path, sem1, scale = False)
            tf.keras.utils.save_img(im_path.replace("labelIds.png","labelIds2.png"),image, scale = False)
            

