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
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "datasets","TopAir"]), help="path to folder containing the databases")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "topair"]), help="path to folder to store csv files")
a = parser.parse_args()

#FRAME_SKIP = 1 # Downsample framerate
FRAME_SKIP = [1,2,3,1,2,3,1,3,2,1,2,3,3,2,1]

if __name__== "__main__":

    os.makedirs(a.output_dir, exist_ok=True)

    data = ["Africa_1", "Africa_2", "Africa_3", "CityPark_1", "CityPark_2", "CityPark_3", "CityPark_4", "CityPark_6", "CityPark_8", "OakForest_1", "OakForest_2", "OakForest_3", "RuralAust1_2", "RuralAust3_1", "RuralAust3_2"]
    sensors = [["images", ".png"], ["depth", ".png"], ["seg_id", ".png"]]
    

    for nset, set in enumerate(data):
        print("Processing %s" % (set))
        
        out_dir = a.output_dir
        file_name = os.path.join(out_dir, "traj_%s.csv" % str(nset).zfill(4))
        trans = [0,0,0]
        # Create csv file
        with open(file_name, 'w') as file:
            file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera_l", "depth","semantic", "qw", "qx", "qy", "qz", "tx", "ty", "tz"))
            def get_path(sensor, index, ext):
                im_name = str(index).zfill(6) + "." + ext
                path = os.path.join(*[set, sensor, im_name])
                return path
            
            
            ##
            data = pd.read_csv("/media/DATA_4TB/Yara/TopAir/"+set+"/airsim_rec.txt",sep = "\t")
            #traj_len = len(x)
            traj_len = len(data)//FRAME_SKIP[nset]
            
            
            print("Trajectory Length: ")
            print(traj_len)
            
            index = -1
            camera_l = get_path("images", 0, "png")
            depth = get_path("depth", 0, "png")
            seg = get_path("seg_id", 0, "png")
            quat = [0.0, 0.0, 0.0, 1.0]
            trans = [0.0001, 0.0001, 0.0001]
            file.write("%i\t%s\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, camera_l, depth, seg, quat[3], quat[0], quat[1], quat[2], trans[0], trans[1], trans[2]))
            
            # Iterate over sequence samples
            for index in range(traj_len-1):
                # Compute frame-to-frame camera motion
                i = index*FRAME_SKIP[nset]
                ##
                #f1 = open(os.path.join(*[a.db_path, set, 'metadata',str(i).zfill(6)+".json"]))
                #data1 = json.load(f1)
                #f2 = open(os.path.join(*[a.db_path, set, 'metadata',str(i+FRAME_SKIP).zfill(6)+".json"]))
                #data2 = json.load(f2)
                ##
                # get rotation and translation between two consecutive frames
                q1 = np.array([data.loc[i, 'Q_X'], data.loc[i, 'Q_Y'], data.loc[i, 'Q_Z'], data.loc[i, 'Q_W']])
                q2 = np.array([data.loc[i+FRAME_SKIP[nset], 'Q_X'], data.loc[i+FRAME_SKIP[nset], 'Q_Y'], data.loc[i+FRAME_SKIP[nset], 'Q_Z'], data.loc[i+FRAME_SKIP[nset], 'Q_W']])
                r1 = R.from_quat(q1)
                r2 = R.from_quat(q2)
                r1 = r1.as_matrix()
                r2 = r2.as_matrix()
                
                p1 = np.array([data.loc[i, 'POS_X'], data.loc[i, 'POS_Y'], data.loc[i, 'POS_Z']])
                p2 = np.array([data.loc[i+FRAME_SKIP[nset], 'POS_X'], data.loc[i+FRAME_SKIP[nset], 'POS_Y'], data.loc[i+FRAME_SKIP[nset], 'POS_Z']])
                
                # rotation of camera frame
                r = R.from_euler('yzx',[1.57, 0, 1.57])
                r = r.as_matrix()
                
                r11 = np.matmul(r.transpose(), r1.transpose())
                r22 = np.matmul(r2, r)
                
                ##
                
                #trans = np.matmul(r1.transpose(),p2-p1)
                #rot_mat = np.matmul(r1.transpose(),r2)
                
                #trans_b = trans
                trans = np.matmul(r11,p2-p1)
                rot_mat = np.matmul(r11,r22)
                rot = R.from_matrix(rot_mat)
                quat = rot.as_quat()
                
                #if trans[0] == 0 and trans[1] == 0 and trans[2] == 0 and index > 0:
                if trans[0] == 0 and trans[1] == 0 and trans[2] == 0:
                    trans[0] = 0.0001
                    trans[1] = 0.0001
                    trans[2] = 0.0001
                    #trans[0] = trans_b[0]
                    #trans[1] = trans_b[1]
                    #trans[2] = trans_b[2]
                
                camera_l = get_path("images", i+FRAME_SKIP[nset], "png")
                depth = get_path("depth", i+FRAME_SKIP[nset], "png")
                seg = get_path("seg_id", i+FRAME_SKIP[nset], "png")
                
                # Write sample to file
                file.write("%i\t%s\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, camera_l, depth, seg, quat[3], quat[0], quat[1], quat[2], trans[0], trans[1], trans[2]))

