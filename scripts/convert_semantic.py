import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import tensorflow as tf
import time
from PIL import Image

class_index = {
        0: [(127, 175, 230), 'Sky'],
        1: [(75, 163, 185),'water plane'],
        2: [(50, 128, 0),  'Trees'],
        3: [(117, 97, 97),  'Dirt Ground'],
        4: [(53, 94, 59), 'Ground vegetation'],   
        5: [(70, 70, 70),   'Rocks'],
        6: [(128, 64, 128),  'Road'],
        7: [(64, 64, 128),  'man-made construction'],
        8: [(128, 64, 64),  'others']
        }
paths = ["/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/seg_000004.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/seg_000008.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3010/seg_000060.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3010/seg_000064.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4000/seg_000004.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4000/seg_000008.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4011/seg_000192.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4011/seg_000196.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2001/seg_000004.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2001/seg_000008.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2022/seg_000540.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2022/seg_000544.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2029/seg_000016.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2029/seg_000020.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/seg_000016.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/seg_000020.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5023/seg_000260.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5023/seg_000264.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0000/seg_000400.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0000/seg_000404.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0011/seg_000072.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0011/seg_000076.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1019/seg_000320.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1019/seg_000324.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1025/seg_000048.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1025/seg_000052.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6007/seg_000348.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6007/seg_000352.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6020/seg_000016.PNG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6020/seg_000020.PNG"]


for img in paths:
    input_img = cv2.imread(img)
    x1 = np.copy(input_img[:,:,0])
    x2 = np.copy(input_img[:,:,0])
    x3 = np.copy(input_img[:,:,0])
    
    for key in class_index:
        #print(key)
        x1[x1 == key] = class_index[key][0][0]
        x2[x2 == key] = class_index[key][0][1]
        x3[x3 == key] = class_index[key][0][2]
        
    x1 = np.expand_dims(x1, axis = -1)
    x2 = np.expand_dims(x2, axis = -1)
    x3 = np.expand_dims(x3, axis = -1)
    
    img_seg = np.append(x1,x2, axis = 2)
    img_seg = np.append(img_seg,x3, axis = 2)
    
    im = Image.fromarray(img_seg.astype(np.uint8))
    im.save(img.replace("seg","segg"))
'''
input_img = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/test3/input_img_454_192_0.png")

x1 = np.copy(input_img[:,:,0])
x2 = np.copy(input_img[:,:,0])
x3 = np.copy(input_img[:,:,0])

for key in class_index:
    print(key)
    x1[x1 == key] = class_index[key][0][0]
    x2[x2 == key] = class_index[key][0][1]
    x3[x3 == key] = class_index[key][0][2]
    
x1 = np.expand_dims(x1, axis = -1)
x2 = np.expand_dims(x2, axis = -1)
x3 = np.expand_dims(x3, axis = -1)

img_seg = np.append(x1,x2, axis = 2)
img_seg = np.append(img_seg,x3, axis = 2)

im = Image.fromarray(img_seg.astype(np.uint8))
im.save("/home/yara/drone_depth/Semantic_M4Depth/scripts/test3/seg_input_img_454_192_0.png")
'''
