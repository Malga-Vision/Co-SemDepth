import tensorflow as tf
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def convert_idx(T, K_mat):
    K_inv = tf.linalg.inv(K_mat)
    conversion = tf.matmul(K_mat, tf.matmul(T,K_inv))
    return conversion



qw_test = 0.999952
qx_test = -0.006936
qy_test = 0.006835
qz_test = -0.000463
tx_test = -0.268941
ty_test = -0.319521
tz_test = 1.550722

fx_test = 96.
fy_test = 96.
cx_test = 96.
cy_test = 96.

h_test = 192
w_test = 192


qw = 0.999
qx = 0.001205
qy = 0.000034
qz = -0.000772
tx = 0.454566
ty = -0.237295
tz = 2.806879
'''
fx = 512
fy = 512
cx = 512
cy = 512

h = 1024
w = 1024
'''
fx = 192
fy = 192
cx = 192
cy = 192
h = 384
w = 384

zpos = 30

q = tf.convert_to_tensor([qx_test,qy_test,qz_test,qw_test])
trans = tf.convert_to_tensor([tx_test,ty_test,tz_test])

#q = tf.convert_to_tensor([qx,qy,qz,qw])
#trans = tf.convert_to_tensor([tx,ty,tz])

r = R.from_quat(q)
rot_mat = r.as_matrix()

T1 = tf.convert_to_tensor([[rot_mat[0,0], rot_mat[0,1], rot_mat[0,2], trans[0]],[rot_mat[1,0], rot_mat[1,1], rot_mat[1,2], trans[1]],[rot_mat[2,0], rot_mat[2,1], rot_mat[2,2], trans[2]],[0,0,0,1]])

T = T1

K = tf.convert_to_tensor([[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0],[0,0,0,1]], dtype = tf.float32)
conv_mat = convert_idx(T, K)

'''
file = tf.io.read_file("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/000008.PNG")
file = tf.io.read_file("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/000020.PNG")
image = tf.image.decode_png(file, dtype=tf.uint16)
image = tf.bitcast(image, tf.float16)
depth = 512./tf.cast(image, dtype=tf.float32)
depth = tf.image.resize(depth, [h,w])
#depth = depth.numpy()
'''
depth = np.load("/home/yara/drone_depth/Semantic_M4Depth/scripts/test_unknown_others/depth_192_1.npy")
depth = tf.convert_to_tensor(depth)



one_mat = tf.ones([1,h,w])
xn = tf.range(w)
yn = tf.range(h)
'''
one_mat = tf.ones([1,h,w])
xn = tf.range(w)
yn = tf.range(h)
'''
xv, yv = tf.meshgrid(xn,yn)
xv = tf.cast(xv, dtype= tf.float32)
yv = tf.cast(yv, dtype= tf.float32)

xv = tf.math.multiply(tf.squeeze(depth),xv)
yv = tf.math.multiply(tf.squeeze(depth),yv)
one_mat_d = tf.math.multiply(tf.squeeze(depth),tf.squeeze(one_mat))
one_mat_d = tf.expand_dims(one_mat_d, axis = 0)


xv = tf.expand_dims(xv, axis = 0)
yv = tf.expand_dims(yv, axis = 0)
grid = tf.concat([xv, yv], axis=0)
#input_mat = zpos*tf.concat([grid, one_mat], axis = 0)
input_mat = tf.concat([grid, one_mat_d], axis = 0)
input_mat = tf.concat([input_mat, one_mat], axis = 0)
input_mat = tf.transpose(input_mat, perm = [1,2,0])
input_mat = tf.expand_dims(input_mat, axis = -1)

mapped = tf.matmul(conv_mat, input_mat)

x = tf.math.round(mapped[:,:,0]/mapped[:,:,2])
y = tf.math.round(mapped[:,:,1]/mapped[:,:,2])

x = tf.cast(x, tf.int32)
y = tf.cast(y, tf.int32)

x = tf.squeeze(x)
y = tf.squeeze(y)

#x = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, x)
#y = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, y)

x2 = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, x)
y2 = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), h, y)

#input_img = cv2.imread("/media/DATA_4TB/Yara/midair/Kite_training/cloudy/color_left/trajectory_3001/000004.JPEG")
#input_img = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/test_unknown_others/seg_input_img_192_1.png")
input_img = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/000016.JPEG")
input_img = tf.convert_to_tensor(input_img)
input_img = tf.image.resize(input_img, [h,w])

others_tensor = tf.ones([1,w,3])
input_img_upd = tf.concat([input_img, others_tensor], axis = 0)

print(input_img_upd)
out1 = tf.gather(input_img_upd, y2)
out2 = tf.gather(out1, x2, batch_dims = 2)

cv2.imwrite("converted_img_tf.png",out2.numpy())









