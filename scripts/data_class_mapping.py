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

tartanair_class = {
    75: 5,
    108: 4,
    112: 0,
    133: 5,
    145: 5,
    151: 4,
    152: 4,
    205: 3,
    218: 5, 
    219: 4,
    232: 4,
    234: 5,
    240: 4,
    241: 5,
    250: 4
}

data = ["Kite_training","PLE_training"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","midair"])

'''
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
'''

def map_wuav(a):
    #print(a)
    if (a == [255,255,0]).all():
        return 0
    elif (a == [0,127,0]).all():
        return 2
    elif (a == [69,132,19]).all():
        return 2
    elif (a == [65,53,0]).all():
        return 3
    elif (a == [0,76,130]).all():
        return 3
    elif (a == [152,251,152]).all():
        return 3
    elif (a == [171,126,151]).all():
        return 5
    elif (a == [255,0,0]).all():
        return 1
    elif (a == [0,150,250]).all():
        return 7
    elif (a == [195,176,115]).all():
        return 8
    elif (a == [128,64,128]).all():
        return 6
    elif (a == [228,77,255]).all():
        return 6
    elif (a == [123,123,123]).all():
        return 4
    elif (a == [255,255,255]).all():
        return 4
    elif (a == [0,0,200]).all():
        return 8
    elif (a == [0,0,0]).all():
        return 8
    else:
        return 8

#data = ["Seq000","Seq001","Seq002","Seq003"]
data = ["Seq003"]
db_path = os.path.join(*["/media/DATA_4TB/Yara","wildUAV"])

for traj in data:
    print("Processing %s" % (traj))
    imgs = os.listdir(os.path.join(*[db_path, traj, 'segmentation']))
    for img in imgs:
        print(img)
        inp_path = os.path.join(*[db_path, traj, 'segmentation', img])
        out_path = os.path.join(*[db_path, traj, 'seg_upd', img])
        im = cv2.imread(inp_path)
        im_upd = np.apply_along_axis(map_wuav, -1,im)
        cv2.imwrite(out_path,im_upd)


'''

## For Gascola
data = ["Easy","Hard"]

dir_path = os.path.dirname(os.path.realpath(__file__))
db_path = os.path.join(*[dir_path,"..", "datasets","TartanAir","gascola","gascola"])
s2 = set(list(tartanair_class.keys()))
for level in data:
    trajectories = os.listdir(os.path.join(db_path,level))
    for traj in trajectories:
        print("Processing %s %s" % (level, traj))
        #out_dir = os.path.join(*[db_path, level, traj, 'seg_upd'])
        #os.makedirs(out_dir, exist_ok=True)
        imgs = os.listdir(os.path.join(*[db_path, level, traj, 'seg_upd']))
        traj_len = len(imgs)
        for img in imgs:
            #print(img)
            inp_path = os.path.join(*[db_path, level, traj, 'seg_upd', img])
            out_path = os.path.join(*[db_path, level, traj, 'seg_upd', img])
            img_np = np.load(inp_path)
            values = np.unique(img_np)
            s1 = set(values.tolist())
            missing = s1 - s2
            
            if len(missing) > 0:
                print(img)
                print(missing)
            else:
                img_upd = np.vectorize(tartanair_class.get)(img_np)
                img_upd = img_upd.astype('uint8')
                with open(out_path, 'wb') as f:
                    np.save(out_path, img_upd)
           
'''



             
                
                
                


