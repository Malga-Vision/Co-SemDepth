import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import tensorflow as tf
import time


start = time.time()
def convert_idx(T, K_mat):
    K_inv = np.linalg.inv(K_mat)
    #conversion = np.dot(K_mat,np.dot(T,K_inv))
    conversion = np.matmul(K_mat,np.matmul(T,K_inv))
    return conversion

zpos = 100

'''
qw_test = 0.999996
qx_test = 0.000243
qy_test = 0.000257
qz_test = -0.002703
tx_test = 0.252823
ty_test = 1.559872
tz_test = 3.427604

fx_test = 96.
fy_test = 96.
cx_test = 96.
cy_test = 96.

h_test = 192
w_test = 192
'''
paths = ["/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/000004.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3010/000060.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4000/000004.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4011/000192.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2001/000004.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2022/000540.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2029/000016.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/000016.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5023/000260.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0000/000400.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0011/000072.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1019/000320.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1025/000048.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6007/000348.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6020/000016.JPEG"]
depth_paths = ["/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/000008.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3010/000064.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4000/000008.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4011/000196.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2001/000008.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2022/000544.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2029/000020.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/000020.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5023/000264.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0000/000404.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0011/000076.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1019/000324.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1025/000052.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6007/000352.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6020/000020.PNG"]
quat = [[0.001205,0.000034, -0.000772, 0.999],
     [0.004707, 0.02056, 0.019982, 0.999578],
     [0.00006, -0.00003, -0.000382, 1],
     [0.017447, 0.008413, 0.029066, 0.99939],
     [0.001205, 0.000034, -0.000772, 0.999],
     [-0.001166, 0.003796, 0.008215, 0.999958],
     [-0.001009, 0.007594, 0.002089, 0.999968],
     [-0.004843, -0.000736, -0.0601, 0.99818],
     [0.002937, -0.029752, -0.012174, 0.999479],
     [0.000248, 0.001039, -0.000497, 0.999999],
     [0.000594, 0.000013, 0.000151, 1],
     [0.005059, 0.002838, 0.000791, 0.999983],
     [0.003963, 0.00434, 0.010838, 0.999924],
     [0.000251, -0.003681, -0.000807, 0.999993],
     [0.002649, 0.006151, 0.001928, 0.999976]]
t = [[0.454566, -0.237295, 2.806879],
     [-0.809309,-0.430736, 2.368358],
     [0.08351, 0.011731, 1.28624],
     [-0.513386, 0.192791, 0.982601],
     [0.454566, -0.237295, 2.806879],
     [0.263649, -0.388047, 1.627853],
     [-0.0877, 0.186042, 0.72677],
     [0.248859, 0.525718, 0.829297],
     [1.359945, 0.001425, -0.079429],
     [0.372412, -0.833617, 3.096381],
     [-0.30333, -0.554263, 1.846618],
     [-0.303405, -0.48115, 1.966117],
     [-0.154719, -0.038303, 1.304788],
     [0.264417, 0.175466, 1.311091],
     [0.281628, 0.007867, 1.23846]]




'''
qw = 0.9999
qx = -0.000855
qy = 0.000671
qz = -0.00053
tx = 0.37
ty = -4.5
tz = 0.0633

h = 3956
w = 5280
'''

h = 384
w = 384
fx = 192
fy = 192
cx = 192
cy = 192

'''
fx = 512
fy = 512
cx = 512
cy = 512
'''
'''
fx = 4551.4
fy = 4550.6
cx = 2446.336
cy = 1966.132
'''
'''
qx = 0.001205
qy= 0.000034
qz = -0.000772
qw = 0.999

tx = 0.454566
ty = -0.237295
tz = 2.806879
'''
'''
qw = 0.99818
qx = -0.004843
qy = -0.000736
qz = -0.0601
tx = 0.248859
ty = 0.525718
tz = 0.829297

#q = [qx_test, qy_test, qz_test, qw_test]
#trans = [tx_test, ty_test, tz_test]

q = [qx, qy, qz, qw]
trans = [tx, ty, tz]

r = R.from_quat(q)
rot_mat = r.as_matrix()
T1 = np.array([[rot_mat[0,0], rot_mat[0,1], rot_mat[0,2], trans[0]],[rot_mat[1,0], rot_mat[1,1], rot_mat[1,2], trans[1]], [rot_mat[2,0], rot_mat[2,1], rot_mat[2,2], trans[2]], [0,0,0,1]])
#T = np.linalg.inv(T1)
T = T1
    
    
K = np.array([[fx, 0, cx, 0],[0, fy, cy, 0],[0, 0, 1, 0], [0, 0, 0, 1]])
#K = np.array([[fx, 0, cx, 0],[0, fy, cy, 0],[0, 0, 1, 0], [0, 0, 0, 1]])
    
conv_mat = convert_idx(T, K)
#print("Conversion Matrix: ",conv_mat)

#file = tf.io.read_file("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/000008.PNG")
file = tf.io.read_file("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/000020.PNG")
image = tf.image.decode_png(file, dtype=tf.uint16)
image = tf.bitcast(image, tf.float16)
depth = 512./tf.cast(image, dtype=tf.float32)
depth = tf.image.resize(depth, [h,w])
depth = depth.numpy()

#depth = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/test/depth_img_192_0.png")
#depth = np.load("/home/yara/drone_depth/Semantic_M4Depth/scripts/test3_unknown_others/depth_192_2.npy")

one_mat = np.ones([1,h,w])
xn = np.arange(w)
yn = np.arange(h)
#one_mat = np.ones([1,h,w])
#xn = np.arange(w)
#yn = np.arange(h)
xv, yv = np.meshgrid(xn,yn)
###

xv = np.multiply(np.squeeze(depth),xv)
yv = np.multiply(np.squeeze(depth),yv)
one_mat_d = np.multiply(np.squeeze(depth),np.squeeze(one_mat))
one_mat_d = np.expand_dims(one_mat_d, axis = 0)

    
###
xv = np.expand_dims(xv, axis = 0)
yv = np.expand_dims(yv, axis = 0)
    
grid = np.append(xv, yv, axis=0)
#input_mat = zpos*np.append(grid, one_mat, axis = 0)
input_mat = np.append(grid, one_mat_d, axis = 0)
input_mat = np.append(input_mat, one_mat, axis = 0)
input_mat = np.moveaxis(input_mat, 0 ,-1)
input_mat = np.expand_dims(input_mat, axis = -1)
    
#print("Input Matrix: ",input_mat)
mapped = np.matmul(conv_mat,input_mat)
mapped[:,:,0] = np.round(mapped[:,:,0]/mapped[:,:,2])
mapped[:,:,1] = np.round(mapped[:,:,1]/mapped[:,:,2])

mapped = np.squeeze(mapped.astype(int))
    
#input_img = cv2.imread("/media/DATA_4TB/Yara/wildUAV/Seq001/img/000002.png")
#input_img = cv2.imread("/media/DATA_4TB/Yara/midair/Kite_training/cloudy/color_left/trajectory_3001/000004.JPEG")
#input_img = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/test3_unknown_others/seg_input_img_192_2.png")
input_img = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/000016.JPEG")
input_img = cv2.resize(input_img, (h,w), interpolation = cv2.INTER_LINEAR)
input_img[0,0,0] = 0
input_img[0,0,1] = 0
input_img[0,0,2] = 0

cond = (mapped[:,:,0] >= w) | (mapped[:,:,0] < 0) | (mapped[:,:,1] >= h) | (mapped[:,:,1] < 0)

mapped[cond] = [0,0,0,0]
    
output_img = input_img[mapped[:,:,1],mapped[:,:,0],:]
    
#output_img = np.zeros([h,w,3])
#output_img[mapped[:,:,1],mapped[:,:,0],:] = input_img
    
end = time.time()
print("Inference time = ", end-start)
    
cv2.imwrite("/home/yara/drone_depth/Semantic_M4Depth/scripts/converted_img.png",output_img)



'''
for i in range(15):
    q = quat[i]
    trans = t[i]
    r = R.from_quat(q)
    rot_mat = r.as_matrix()
    T1 = np.array([[rot_mat[0,0], rot_mat[0,1], rot_mat[0,2], trans[0]],[rot_mat[1,0], rot_mat[1,1], rot_mat[1,2], trans[1]], [rot_mat[2,0], rot_mat[2,1], rot_mat[2,2], trans[2]], [0,0,0,1]])
    #T = np.linalg.inv(T1)
    T = T1
    
    
    
    K = np.array([[fx, 0, cx, 0],[0, fy, cy, 0],[0, 0, 1, 0], [0, 0, 0, 1]])
    
    conv_mat = convert_idx(T, K)
    #print("Conversion Matrix: ",conv_mat)
    
    #file = tf.io.read_file("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/000004.PNG")
    file = tf.io.read_file(depth_paths[i])
    image = tf.image.decode_png(file, dtype=tf.uint16)
    image = tf.bitcast(image, tf.float16)
    depth = 512./tf.cast(image, dtype=tf.float32)
    depth = tf.image.resize(depth, [h,w])
    depth = depth.numpy()
    
    
    one_mat = np.ones([1,h,w])
    xn = np.arange(w)
    yn = np.arange(h)
    xv, yv = np.meshgrid(xn,yn)
    ###
    
    xv = np.multiply(np.squeeze(depth),xv)
    yv = np.multiply(np.squeeze(depth),yv)
    one_mat_d = np.multiply(np.squeeze(depth),np.squeeze(one_mat))
    one_mat_d = np.expand_dims(one_mat_d, axis = 0)
    
    ###
    xv = np.expand_dims(xv, axis = 0)
    yv = np.expand_dims(yv, axis = 0)
    
    grid = np.append(xv, yv, axis=0)
    #input_mat = zpos*np.append(grid, one_mat, axis = 0)
    input_mat = np.append(grid, one_mat_d, axis = 0)
    input_mat = np.append(input_mat, one_mat, axis = 0)
    input_mat = np.moveaxis(input_mat, 0 ,-1)
    input_mat = np.expand_dims(input_mat, axis = -1)
    
    #print("Input Matrix: ",input_mat)
    mapped = np.matmul(conv_mat,input_mat)
    mapped[:,:,0] = np.round(mapped[:,:,0]/mapped[:,:,2])
    mapped[:,:,1] = np.round(mapped[:,:,1]/mapped[:,:,2])
    
    mapped = np.squeeze(mapped.astype(int))
    
    #input_img = cv2.imread("/media/DATA_4TB/Yara/wildUAV/Seq001/img/000002.png")
    #input_img = cv2.imread("/media/DATA_4TB/Yara/midair/Kite_training/cloudy/color_left/trajectory_3001/000004.JPEG")
    input_img = cv2.imread((paths[i].replace("/000","/segg_000")).replace(".JPEG",".PNG"))
    input_img = cv2.resize(input_img, [h,w], interpolation=cv2.INTER_LINEAR)
    input_img[0,0,0] = 0
    input_img[0,0,1] = 0
    input_img[0,0,2] = 0
    
    cond = (mapped[:,:,0] >= w) | (mapped[:,:,0] < 0) | (mapped[:,:,1] >= h) | (mapped[:,:,1] < 0)
    
    mapped[cond] = [0,0,0,0]
    
    output_img = input_img[mapped[:,:,1],mapped[:,:,0],:]
    
    #output_img = np.zeros([h,w,3])
    #output_img[mapped[:,:,1],mapped[:,:,0],:] = input_img
    
    end = time.time()
    print("Inference time = ", end-start)
    
    cv2.imwrite("converted_img_d_seg_"+str(i)+".png",output_img)




