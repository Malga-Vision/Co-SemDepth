'''
Note: This script requires two specific libraries:
    * h5py for opening Mid-Air data records
    * pyquaternion for quaternion operations
Both can be installed with pip:
$ pip install pyquaternion h5py
'''

import os
import argparse
import h5py
#from pyquaternion import Quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import numpy as np
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "datasets","cityscapes"]), help="path to folder containing the databases")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "cityscapes"]), help="path to folder to store csv files")
a = parser.parse_args()

FRAME_SKIP = 1 # Downsample framerate
#FRAME_SKIP = [1,2,3,1,2,3,1]

if __name__== "__main__":

    os.makedirs(a.output_dir, exist_ok=True)
    
    sensors = [["images", ".png"], ["segmentation", ".png"]]
    print("DB path:")
    print(a.db_path)
    envs = os.listdir(os.path.join(a.db_path,'test/images'))
    print(envs)
    for nset, env in enumerate(envs):
        #env = set.split('\\')
        #env = env[-1]
        print("Processing %s" % (env))
        set = os.path.join(a.db_path,'test/images/'+env)
        sett = 'test/images'
        print(set)
        out_dir = a.output_dir
        file_name = os.path.join(out_dir, "%s.csv" % env)
        
        # Create csv file
        with open(file_name, 'w') as file:
            file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera_l", "semantic", "qw", "qx", "qy", "qz", "tx", "ty", "tz"))
            '''
            def get_path(sensor, index, ext, img_type):
                if img_type == "img":
                    im_name = sensor +"_" + str(index).zfill(6) + "_000019_leftImg8bit" "." + ext
                    path = os.path.join(*[sett, sensor, im_name])
                else:
                    im_name = sensor +"_" + str(index).zfill(6) + "_000019_gtFine_labelIds" "." + ext
                    path = os.path.join(*[sett.replace("images","segmentation"), sensor, im_name])
                return path
            '''
            
            ##
            imgs = os.listdir(set)
            #traj_len = len(x)
            traj_len = len(imgs)//FRAME_SKIP
            
            
            print("Trajectory Length: ")
            print(traj_len)
            

            # Iterate over sequence samples
            for index in range(traj_len):
                # Compute frame-to-frame camera motion
                i = index*FRAME_SKIP
                
                ## put dummy data in transition and rotation fields
                trans = [5.0, 5.0, 5.0]
                quat = [5.0, 5.0, 5.0, 5.0]
                
                camera_l = os.path.join(*[sett, env, imgs[index]])
                semantic = (camera_l.replace("images","segmentation")).replace("leftImg8bit", "gtFine_labelIds")
                
                # Write sample to file
                file.write("%i\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, camera_l, semantic, quat[3], quat[0], quat[1], quat[2], trans[0], trans[1], trans[2]))

