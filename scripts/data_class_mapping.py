import numpy as np
import os
import cv2


midair_class = {
    0: 0,
    1: 6,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 4,
    7: 6,
    8: 1,
    9: 6,
    10: 5,
    11: 6,
    12: 6,
    13: 6
}


data = ["Kite_training","PLE_training"]
db_path = os.path.join(*["../datasets","MidAir"])


##For MidAir
for set in data:
    climates = os.listdir(os.path.join(db_path,set))
    for climate in climates:
        print("Processing %s %s" % (set, climate))
        trajectories = os.listdir(os.path.join(*[db_path, set, climate, 'segmentation']))
        for traj_num, (traj) in enumerate(trajectories):
            print(traj)
            imgs = os.listdir(os.path.join(*[db_path, set, climate, 'segmentation', traj]))
            traj_len = len(imgs)
            for img in imgs:
                inp_path = os.path.join(*[db_path, set, climate, 'segmentation', traj, img])
                #print("Img: ", inp_path)
                img_cv = cv2.imread(inp_path)
                img_s = cv2.resize(img_cv, (384,384), interpolation= cv2.INTER_NEAREST)
                img_upd = np.vectorize(midair_class.get)(img_s)
                cv2.imwrite(inp_path, img_upd)
