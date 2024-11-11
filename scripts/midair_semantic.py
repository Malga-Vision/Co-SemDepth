import os
import argparse
import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "data","midair","test_data"]), help="path to folder containing the csv files")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "midair_semantic"]), help="path to folder to store output csv files")
a = parser.parse_args()

if __name__== "__main__":

    os.makedirs(a.output_dir, exist_ok=True)
    data = ["Kite_training", "PLE_training"]
    #seasons1 = ["cloudy","foggy", "sunny", "sunset"]
    #seasons2 = ["fall", "spring", "winter"]
    
    for set in data:
        climates = os.listdir(os.path.join(a.db_path,set))
        for climate in climates:
            trajectories = os.listdir(os.path.join(*[a.db_path, set, climate]))
            for t in trajectories:
                df = pd.read_csv(os.path.join(*[a.db_path, set, climate, t]), sep = '\t')
                #print(df.head())
                f_name = os.path.join(*[a.db_path, set, climate, t])
                with open(f_name, 'w') as file:
                    file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera_l", "disp", "semantic", "qw", "qx", "qy", "qz", "tx", "ty", "tz"))
                    for index, row in df.iterrows():
                        index = row['id']
                        camera_l = row['camera_l']
                        stereo_disp = row['disp']
                        semantic = stereo_disp.replace("stereo_disparity","segmentation")
                        qw = row['qw']
                        qx = row['qx']
                        qy = row['qy']
                        qz = row['qz']
                        tx = row['tx']
                        ty = row['ty']
                        tz = row['tz']
                        file.write("%i\t%s\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, camera_l, stereo_disp, semantic, qw, qx, qy, qz, tx, ty, tz))
        

