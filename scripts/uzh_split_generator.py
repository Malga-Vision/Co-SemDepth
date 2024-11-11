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
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "datasets","UZH"]), help="path to folder containing the databases")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "uzh"]), help="path to folder to store csv files")
a = parser.parse_args()

FRAME_SKIP = 3 # Downsample framerate

if __name__== "__main__":

    os.makedirs(a.output_dir, exist_ok=True)

    data = ["traj_0000", "traj_0001", "traj_0002"]
    sensors = [["img", ".png"]]
    

    for nset, set in enumerate(data):
        print("Processing %s" % (set))
        
        out_dir = a.output_dir
        file_name = os.path.join(out_dir, "traj_%s.csv" % str(nset).zfill(4))

        # Create csv file
        with open(file_name, 'w') as file:
            file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera_l", "qw", "qx", "qy", "qz", "tx", "ty", "tz"))
            def get_path(sensor, index, ext):
                im_name = str(index).zfill(6) + "." + ext
                path = os.path.join(*[set, sensor, im_name])
                return path
            
            
            img_csv = pd.read_csv("/media/DATA_4TB/Yara/UZH/"+set+"/img.csv")
            loc_csv = pd.read_csv("/media/DATA_4TB/Yara/UZH/"+set+"/loc.csv")
            
            #traj_len = len([name for name in os.listdir(os.path.join(*[a.db_path, set, 'metadata']))])
            traj_len = img_csv.shape[0]
            print("Trajectory Length: ")
            print(traj_len)
            
            
            # Iterate over sequence samples
            for index in range(0,traj_len-FRAME_SKIP,FRAME_SKIP):
                # Compute frame-to-frame camera motion
                i = index
                ##
                #f1 = open(os.path.join(*[a.db_path, set, 'metadata',str(i).zfill(6)+".json"]))
                #data1 = json.load(f1)
                #f2 = open(os.path.join(*[a.db_path, set, 'metadata',str(i+FRAME_SKIP).zfill(6)+".json"]))
                #data2 = json.load(f2)
                img1 = set + "/" + img_csv.iloc[i]['image_name']
                img2 = set + "/" + img_csv.iloc[i+FRAME_SKIP]['image_name']
                
                qx1 = loc_csv.iloc[i]['qx']
                qy1 = loc_csv.iloc[i]['qy']
                qz1 = loc_csv.iloc[i]['qz']
                qw1 = loc_csv.iloc[i]['qw']
                qx2 = loc_csv.iloc[i+FRAME_SKIP]['qx']
                qy2 = loc_csv.iloc[i+FRAME_SKIP]['qy']
                qz2 = loc_csv.iloc[i+FRAME_SKIP]['qz']
                qw2 = loc_csv.iloc[i+FRAME_SKIP]['qw']
                tx1 = loc_csv.iloc[i]['tx']
                ty1 = loc_csv.iloc[i]['ty']
                tz1 = loc_csv.iloc[i]['tz']
                tx2 = loc_csv.iloc[i+FRAME_SKIP]['tx']
                ty2 = loc_csv.iloc[i+FRAME_SKIP]['ty']
                tz2 = loc_csv.iloc[i+FRAME_SKIP]['tz']
                
                
                ##
                p1 = np.array([tx1, ty1, tz1])
                p2 = np.array([tx2, ty2, tz2])
                q1 = np.array([qx1, qy1, qz1, qw1])
                q2 = np.array([qx2, qy2, qz2, qw2])
                r1 = R.from_quat(q1)
                r2 = R.from_quat(q2)
                
                r1 = r1.as_matrix()
                r2 = r2.as_matrix()
                
                ##
                # get rotation and translation between two consecutive frames
                #r1 = np.asarray(data1['rotation']).transpose()
                #r2 = np.asarray(data2['rotation']).transpose()
                #p1 = np.asarray(data1['translation'])
                #p2 = np.asarray(data2['translation'])
                
                trans = np.matmul(r1.transpose(),p2-p1)
                rot_mat = np.matmul(r1.transpose(),r2)
                rot = R.from_matrix(rot_mat)
                quat = rot.as_quat()
                #t = p2[2] - p1[2]
                
                
                
                #camera_l = get_path("img", i+FRAME_SKIP, "png")
                #depth = get_path("depth", i+FRAME_SKIP, "npy")
                
                # Write sample to file
                file.write("%i\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, img2, quat[3], quat[0], quat[1], quat[2], trans[0], trans[1], trans[2]))

