import tensorflow as tf
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import tensorflow_graphics.geometry.transformation as tfg
import time


counter = 0

def convert_idx(T, K_mat):
    K_inv = tf.linalg.inv(K_mat)
    conversion = tf.matmul(K_mat, tf.matmul(T,K_inv))
    return conversion
  

@tf.function
def get_semantic_depth_reproj(prev_semantic_time, curr_depth_time, rot, trans, camera):
    global counter
    """ Computes the reprojection of semantic map as presented in the paper """
    with tf.compat.v1.name_scope("Semantic_Reprojection"):
        
        b, h, w, ch = prev_semantic_time.get_shape().as_list()
        f = camera['f']
        c = camera['c']
        
        zeros = tf.zeros([b])
        ones = tf.ones([b])
        m = tf.stack((f[:,0], zeros, c[:,0], zeros,
                      zeros, f[:,1], c[:,1], zeros, 
                      zeros, zeros, ones, zeros,
                      zeros, zeros, zeros, ones), axis = -1)
        
        K = tf.reshape(m, shape = [b,4,4])
        qw = tf.expand_dims(rot[:,0],-1)
        q = tf.concat([rot[:,1:], qw] , -1)
        rot_mat = tfg.rotation_matrix_3d.from_quaternion(q)
        T = tf.stack((rot_mat[:,0,0], rot_mat[:,0,1], rot_mat[:,0,2], trans[:,0],
                       rot_mat[:,1,0], rot_mat[:,1,1], rot_mat[:,1,2], trans[:,1],
                       rot_mat[:,2,0], rot_mat[:,2,1], rot_mat[:,2,2], trans[:,2],
                       zeros,zeros,zeros,ones), axis = -1)
        T = tf.reshape(T, [b,4,4])
        conv_mat = convert_idx(T, K)
        
        one_mat = tf.ones([1,h,w])
        xn = tf.range(w)
        yn = tf.range(h)
        xm, ym = tf.meshgrid(xn,yn)
        xm = tf.cast(xm, dtype= tf.float32)
        ym = tf.cast(ym, dtype= tf.float32)
        
        xv = tf.math.multiply(tf.squeeze(curr_depth_time[0]),xm)
        yv = tf.math.multiply(tf.squeeze(curr_depth_time[0]),ym)
        one_mat_d = tf.math.multiply(tf.squeeze(curr_depth_time[0]),tf.squeeze(one_mat))
        one_mat_d = tf.expand_dims(one_mat_d, axis = 0)
        
        xv = tf.expand_dims(xv, axis = 0)
        yv = tf.expand_dims(yv, axis = 0)
        grid = tf.concat([xv, yv], axis=0)
        input_mat = tf.concat([grid, one_mat_d], axis = 0)
        input_mat = tf.concat([input_mat, one_mat], axis = 0)
        input_mat = tf.transpose(input_mat, perm = [1,2,0])
        input_mat = tf.expand_dims(input_mat, axis = -1)
        
        
        input_img = prev_semantic_time[0]
        
        mapped = tf.matmul(conv_mat[0],input_mat)
        x = tf.math.round(mapped[:,:,0]/mapped[:,:,2])
        y = tf.math.round(mapped[:,:,1]/mapped[:,:,2])
        x = tf.cast(x, tf.int32)
        y = tf.cast(y, tf.int32)
        x = tf.squeeze(x)
        y = tf.squeeze(y)
        
        x2 = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, x)
        y2 = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), h, y)
        
        
        z = tf.zeros([1,w,8])
        o = tf.ones([1,w,1])
        others_tensor = tf.concat([z,o], axis=2)
        input_img = tf.concat([input_img, others_tensor], axis = 0)
        
        out = tf.gather(input_img, y2)
        output = [tf.gather(out, x2, batch_dims = 2)]
        
        counter = counter + 1
        for i in range(1,b):
            input_img = prev_semantic_time[i]
            
            z = tf.zeros([1,w,8])
            o = tf.ones([1,w,1])
            others_tensor = tf.concat([z,o], axis=2)
            input_img = tf.concat([input_img, others_tensor], axis = 0)
            
            xv = tf.math.multiply(tf.squeeze(curr_depth_time[i]),xm)
            yv = tf.math.multiply(tf.squeeze(curr_depth_time[i]),ym)
            one_mat_d = tf.math.multiply(tf.squeeze(curr_depth_time[i]),tf.squeeze(one_mat))
            one_mat_d = tf.expand_dims(one_mat_d, axis = 0)
            
            xv = tf.expand_dims(xv, axis = 0)
            yv = tf.expand_dims(yv, axis = 0)
            grid = tf.concat([xv, yv], axis=0)
            input_mat = tf.concat([grid, one_mat_d], axis = 0)
            input_mat = tf.concat([input_mat, one_mat], axis = 0)
            input_mat = tf.transpose(input_mat, perm = [1,2,0])
            input_mat = tf.expand_dims(input_mat, axis = -1)
            
            
            mapped = tf.matmul(conv_mat[i],input_mat)
            x = tf.math.round(mapped[:,:,0]/mapped[:,:,2])
            y = tf.math.round(mapped[:,:,1]/mapped[:,:,2])
            x = tf.cast(x, tf.int32)
            y = tf.cast(y, tf.int32)
            x = tf.squeeze(x)
            y = tf.squeeze(y)
            
            x2 = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, x)
            y2 = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), h, y)
            
            
            out = tf.gather(input_img, y2)
            out = [tf.gather(out, x2, batch_dims = 2)]
            output = tf.concat([output, out], axis = 0)
            
            counter = counter + 1
        
    return output

