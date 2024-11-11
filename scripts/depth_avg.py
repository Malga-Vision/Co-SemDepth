import numpy as np
import cv2


depth_paths = ["/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/000008.PNG",
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
               
               

for path in depth_paths:
    depth = cv2.imread(path)
    avg_value = np.mean(depth)
    median_value = np.median(depth)
    print("Depth IMG = ", path)
    print("MEAN = ", avg_value)
    print("MEDIAN = ", median_value)
