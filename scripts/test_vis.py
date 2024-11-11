import tensorflow as tf
from PIL import Image
import numpy as np

class_index = {
    0: [(127, 175, 230), 'Sky'],
    1: [(192, 64, 128),  'Animals'],
    2: [(50, 128, 0),  'Trees'],
    3: [(105, 58, 69),  'Dirt Ground'],
    4: [(53, 94, 59), 'Ground vegetation'],   
    5: [(117, 97, 97),   'Rocky ground'],
    6: [(70,70,70),   'Boulders'],
    7: [(0, 0, 0), 'empty'],
    8: [(75, 163, 185),'water plane'],
    9: [(64, 64, 128),  'man-made construction'],
    10: [(128, 64, 128),  'Road'],
    11: [(192, 64, 128),   'Train track'],
    12: [(128, 128, 192), 'Road sign'],
    13: [(128, 64, 64),  'others']
}

file = tf.io.read_file("/media/DATA_4TB/Yara/midair/Kite_training/sunny/segmentation/trajectory_0001/000033.PNG")
image = tf.image.decode_png(file, dtype = tf.uint8)
img = image.numpy()

x1 = np.copy(img)
x2 = np.copy(img)
x3 = np.copy(img)

for key in class_index:
    x1[x1 == key] = class_index[key][0][0]
    x2[x2 == key] = class_index[key][0][1]
    x3[x3 == key] = class_index[key][0][2]


img_seg = np.append(x1,x2, axis = 2)
img_seg = np.append(img_seg,x3, axis = 2)
print(img_seg)
print(np.shape(img_seg))

im = Image.fromarray(img_seg)
im.save("test_seg.png")




    
    

