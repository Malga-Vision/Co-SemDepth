import numpy as np
import os
import cv2


midair_class = {
    0: 0,
    1: 8,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 5,
    7: 8,
    8: 1,
    9: 7,
    10: 6,
    11: 8,
    12: 8,
    13: 8
}


data = ["Kite_training","PLE_training"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","midair"])


##For MidAir
for set in data:
    climates = os.listdir(os.path.join(db_path,set))
    for climate in climates:
        print("Processing %s %s" % (set, climate))
        out_dir = os.path.join(*[db_path, set, climate, 'segmentation_upd'])
        os.makedirs(out_dir, exist_ok=True)
        trajectories = os.listdir(os.path.join(*[db_path, set, climate, 'segmentation_orig']))
        for traj_num, (traj) in enumerate(trajectories):
            print(traj)
            out_dir = os.path.join(*[db_path, set, climate, 'segmentation_upd', traj])
            os.makedirs(out_dir, exist_ok=True)
            imgs = os.listdir(os.path.join(*[db_path, set, climate, 'segmentation_orig', traj]))
            traj_len = len(imgs)
            for img in imgs:
                if img != 'frames.zip':
                    inp_path = os.path.join(*[db_path, set, climate, 'segmentation_orig', traj, img])
                    out_path = os.path.join(*[db_path, set, climate, 'segmentation_upd', traj, img])
                    print("Img: ", inp_path)
                    img_cv = cv2.imread(inp_path)
                    img_upd = np.vectorize(midair_class.get)(img_cv)
                    cv2.imwrite(out_path, img_upd)


             
                
                
                


