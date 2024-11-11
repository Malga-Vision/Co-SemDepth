import numpy as np
from PIL import Image

class_index = {
    75: [(245, 22, 110), '...'],
    108: [(53, 118, 126),  '...'],
    112: [(88, 207, 100),  '...'],
    133: [(221, 83, 47),  '...'],
    145: [(200, 45, 170), '...'],   
    152: [(17, 91, 237),   '...'],
    205: [(105,127,176),   '...'],
    218: [(131, 182, 184), '...'],
    234: [(21, 119, 12),'...'],
    240: [(106, 43, 131),  '...'],
    250: [(38, 27, 159),  '...'],
    219: [(110, 123, 37),   '...'],
    151: [(159, 58, 173), '...'],
    241: [(44, 216, 103),  '...'],
    232: [(4, 122, 235),  '...']
}

seg = np.load("/media/DATA_4TB/Yara/tartanair/gascola/gascola/Hard/P007/seg_left/000100_left_seg.npy")

x1 = np.copy(seg)
x2 = np.copy(seg)
x3 = np.copy(seg)

x1 = np.expand_dims(x1, axis = -1)
x2 = np.expand_dims(x2, axis = -1)
x3 = np.expand_dims(x3, axis = -1)


for key in class_index:
    x1[x1 == key] = class_index[key][0][0]
    x2[x2 == key] = class_index[key][0][1]
    x3[x3 == key] = class_index[key][0][2]


img_seg = np.append(x1,x2, axis = -1)
img_seg = np.append(img_seg,x3, axis = -1)
            
im = Image.fromarray(img_seg.astype(np.uint8))
im.save("seg_tartan11.png")
