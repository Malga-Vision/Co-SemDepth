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
                img_upd = np.vectorize(midair_class.get)(img_cv)
                cv2.imwrite(inp_path, img_upd)
