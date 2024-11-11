import cv2
import numpy as np

img1_paths = ["/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/000008.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3010/000064.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4000/000008.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4011/000196.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2001/000008.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2022/000544.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2029/000020.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/000020.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5023/000264.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0000/000404.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0011/000076.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1019/000324.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1025/000052.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6007/000352.PNG",
               "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6020/000020.PNG"]


i = 0
for path in img1_paths:
    img1= cv2.imread(path.replace("/000","/segg_000"))
    img1 = cv2.resize(img1, [384,384], interpolation=cv2.INTER_LINEAR)
    img2 = cv2.imread(path[:-10]+"converted_img_d_seg_"+str(i)+".png")
    image1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = (image1_gray.astype("float") - image2_gray.astype("float"))**2
    cv2.imwrite("err_"+str(i)+".png",diff)
    i = i+1

