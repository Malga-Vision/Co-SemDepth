import numpy as np
import os
import cv2
import scipy

median_size = 10

data = ["Kite_training","PLE_training"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","midair"])

##For MidAir
for set in data:
    climates = os.listdir(os.path.join(db_path,set))
    for climate in climates:
        print("Processing %s %s" % (set, climate))
        out_dir = os.path.join(*[db_path, set, climate, 'segmentation_median'])
        os.makedirs(out_dir, exist_ok=True)
        trajectories = os.listdir(os.path.join(*[db_path, set, climate, 'segmentation']))
        for traj_num, (traj) in enumerate(trajectories):
            print(traj)
            out_dir = os.path.join(*[db_path, set, climate, 'segmentation_median', traj])
            os.makedirs(out_dir, exist_ok=True)
            imgs = os.listdir(os.path.join(*[db_path, set, climate, 'segmentation', traj]))
            traj_len = len(imgs)
            for img in imgs:
                if img != 'frames.zip':
                    inp_path = os.path.join(*[db_path, set, climate, 'segmentation', traj, img])
                    out_path = os.path.join(*[db_path, set, climate, 'segmentation_median', traj, img])
                    #print("Img: ", inp_path)
                    img_cv = cv2.imread(inp_pat
                    h)
                    img_upd = np.vectorize(midair_class.get)(img_cv)
                    cv2.imwrite(out_path, img_upd)

