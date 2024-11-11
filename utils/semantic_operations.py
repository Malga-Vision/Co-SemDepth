import tensorflow as tf
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import tensorflow_graphics.geometry.transformation as tfg
import time

zpos = 100

counter = 0

def convert_idx(T, K_mat):
    K_inv = tf.linalg.inv(K_mat)
    conversion = tf.matmul(K_mat, tf.matmul(T,K_inv))
    return conversion
    
    
@tf.function
def get_semantic_reproj(prev_semantic_time, rot, trans, camera):
    """ Computes the reprojection of semantic map as presented in the paper """
    with tf.compat.v1.name_scope("Semantic_Reprojection"):
        print("ROT = ",rot)
        
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
        #r = R.from_quat(q)
        #rot_mat = r.as_matrix()
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
        xv, yv = tf.meshgrid(xn,yn)
        xv = tf.cast(xv, dtype= tf.float32)
        yv = tf.cast(yv, dtype= tf.float32)
        xv = tf.expand_dims(xv, axis = 0)
        yv = tf.expand_dims(yv, axis = 0)
        grid = tf.concat([xv, yv], axis=0)
        input_mat = zpos*tf.concat([grid, one_mat], axis = 0)
        input_mat = tf.concat([input_mat, one_mat], axis = 0)
        input_mat = tf.transpose(input_mat, perm = [1,2,0])
        input_mat = tf.expand_dims(input_mat, axis = -1)
        
        '''
        one_mat = np.ones([1,h,w])
        xn = np.arange(w)
        yn = np.arange(h)
        xv, yv = np.meshgrid(xn,yn)
        xv = np.expand_dims(xv, axis = 0)
        yv = np.expand_dims(yv, axis = 0)
        grid = np.append(xv, yv, axis=0)
        input_mat = zpos*np.append(grid, one_mat, axis = 0)
        input_mat = np.append(input_mat, one_mat, axis = 0)
        input_mat = np.moveaxis(input_mat, 0 ,-1)
        input_mat = np.expand_dims(input_mat, axis = -1)
        #input_mat = np.expand_dims(input_mat, axis = 0)
        #input_mat = np.repeat(input_mat, b, axis = 0)
        input_mat = tf.convert_to_tensor(input_mat, dtype = tf.float32)
        '''
        
        input_img = prev_semantic_time[0]
            
        mapped = tf.matmul(conv_mat[0],input_mat)
        x = tf.math.round(mapped[:,:,0]/mapped[:,:,2])
        y = tf.math.round(mapped[:,:,1]/mapped[:,:,2])
        x = tf.cast(x, tf.int32)
        y = tf.cast(y, tf.int32)
        x = tf.squeeze(x)
        y = tf.squeeze(y)
        x = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, x)
        y = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, y)
                 
        out = tf.gather(input_img, y)
        output = [tf.gather(out, x, batch_dims = 2)]
        
        for i in range(1,b):
            input_img = prev_semantic_time[i]
            
            mapped = tf.matmul(conv_mat[i],input_mat)
            x = tf.math.round(mapped[:,:,0]/mapped[:,:,2])
            y = tf.math.round(mapped[:,:,1]/mapped[:,:,2])
            x = tf.cast(x, tf.int32)
            y = tf.cast(y, tf.int32)
            x = tf.squeeze(x)
            y = tf.squeeze(y)
            x = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, x)
            y = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, y)
            
            out = tf.gather(input_img, y)
            out = [tf.gather(out, x, batch_dims = 2)]
            output = tf.concat([output, out], axis = 0)
        #print(output)
        print("THE END")
    return output

@tf.function
def get_semantic_depth_reproj(prev_semantic_time, curr_depth_time, rot, trans, camera):
    global counter
    """ Computes the reprojection of semantic map as presented in the paper """
    with tf.compat.v1.name_scope("Semantic_Reprojection"):
        #print("ROT = ",rot)
        #print("TRANS = ",trans)
        b, h, w, ch = prev_semantic_time.get_shape().as_list()
        f = camera['f']
        c = camera['c']
        #print("H = ",h)
        #print("W = ", w)
        #print("CH= ",ch)
        #print("F = ",f)
        #print("C = ", c)
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
        #print("DEPTH = ", curr_depth_time[0])
        #print("DEPTH NUMPY = ", curr_depth_time[0].numpy())
        #cv2.imwrite("depth_img_"+str(counter)+"_"+str(h)+"_0.png",curr_depth_time[0].numpy())
        #np.save("depth_"+str(h)+"_0.npy", curr_depth_time[0].numpy())
        xv = tf.expand_dims(xv, axis = 0)
        yv = tf.expand_dims(yv, axis = 0)
        grid = tf.concat([xv, yv], axis=0)
        input_mat = tf.concat([grid, one_mat_d], axis = 0)
        input_mat = tf.concat([input_mat, one_mat], axis = 0)
        input_mat = tf.transpose(input_mat, perm = [1,2,0])
        input_mat = tf.expand_dims(input_mat, axis = -1)
        
        
        input_img = prev_semantic_time[0]
        #input_img_w = tf.expand_dims(tf.math.argmax(input_img, -1), -1)
        #cv2.imwrite("input_img_"+str(h)+"_0.png",input_img_w.numpy())
        mapped = tf.matmul(conv_mat[0],input_mat)
        x = tf.math.round(mapped[:,:,0]/mapped[:,:,2])
        y = tf.math.round(mapped[:,:,1]/mapped[:,:,2])
        x = tf.cast(x, tf.int32)
        y = tf.cast(y, tf.int32)
        x = tf.squeeze(x)
        y = tf.squeeze(y)
        
        x2 = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), 0, x)
        y2 = tf.where((x >= w) | (x < 0) | (y >= h) | (y < 0), h, y)
        
        
        #print("INPUT IMG= ", input_img)
        
        z = tf.zeros([1,w,8])
        o = tf.ones([1,w,1])
        others_tensor = tf.concat([z,o], axis=2)
        input_img = tf.concat([input_img, others_tensor], axis = 0)
        #np.save("input_img_"+str(h)+"_0.npy", input_img.numpy())
        #input_img_w = tf.expand_dims(tf.math.argmax(input_img, -1), -1)
        #cv2.imwrite("input_img_"+str(counter)+"_"+str(h)+"_0.png",input_img_w.numpy())
        
        #print("INPUT IMG UPD= ", input_img)
        
        out = tf.gather(input_img, y2)
        output = [tf.gather(out, x2, batch_dims = 2)]
        #out_w = tf.expand_dims(tf.math.argmax(output[0], -1), -1)
        #cv2.imwrite("output_img_"+str(counter)+"_"+str(h)+"_0.png",out_w.numpy())
        #np.save("output_img_"+str(h)+"_0.npy", output[0].numpy())
        counter = counter + 1
        for i in range(1,b):
            input_img = prev_semantic_time[i]
            
            z = tf.zeros([1,w,8])
            o = tf.ones([1,w,1])
            others_tensor = tf.concat([z,o], axis=2)
            input_img = tf.concat([input_img, others_tensor], axis = 0)
            
            #input_img_w = tf.expand_dims(tf.math.argmax(input_img, -1), -1)
            #cv2.imwrite("input_img_"+str(counter)+"_"+str(h)+"_"+str(i)+".png",input_img_w.numpy())
            #np.save("input_img_"+str(h)+"_"+str(i)+".npy", input_img.numpy())
            xv = tf.math.multiply(tf.squeeze(curr_depth_time[i]),xm)
            yv = tf.math.multiply(tf.squeeze(curr_depth_time[i]),ym)
            one_mat_d = tf.math.multiply(tf.squeeze(curr_depth_time[i]),tf.squeeze(one_mat))
            one_mat_d = tf.expand_dims(one_mat_d, axis = 0)
            #np.save("depth_"+str(h)+"_"+str(i)+".npy", curr_depth_time[i].numpy())
            #cv2.imwrite("depth_img_"+str(counter)+"_"+str(h)+"_"+str(i)+".png",curr_depth_time[i].numpy())
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
            #print("OUTPUT SHAPE = ", output)
            #out_w = tf.expand_dims(tf.math.argmax(out[0], -1), -1)
            #cv2.imwrite("output_img_"+str(counter)+"_"+str(h)+"_"+str(i)+".png",out_w.numpy())
            #np.save("output_img_"+str(h)+"_"+str(i)+".npy", out[0].numpy())
            counter = counter + 1
        #print(output)
    return output

