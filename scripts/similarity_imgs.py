import cv2
from skimage import metrics
import numpy as np

imgs = ["/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/000008.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3010/000064.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4000/000008.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/fall_4011/000196.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2001/000008.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2022/000544.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/foggy_2029/000020.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5001/000020.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/spring_5023/000264.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0000/000404.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunny_0011/000076.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1019/000324.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/sunset_1025/000052.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6007/000352.JPEG",
         "/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/winter_6020/000020.JPEG"]
depths= ["30","50", "100", "d2"]

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

img1 = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/converted_img_s_depth1.png")
img2 = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/cloudy_3001/000008_s.png")

#img1 = cv2.resize(image1, [384,384], interpolation=cv2.INTER_LINEAR)
#img2 = cv2.resize(img2, [384,384], interpolation=cv2.INTER_LINEAR)
        
image1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
print(f"SSIM Score: ", round(ssim_score[0], 2))
        
hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        
metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
print(f"Histogram Similarity Score: ", round(metric_val, 2))
        
mse_value = mse(image1_gray, image2_gray)
print(f"MSE Value: ",round(mse_value,2))
        
'''
for i in range(15):
    path1 = imgs[i]
    print("FOLDER: ", i)
    for d in depths:
        print("DEPTH: ", d)
        path2 = path1[:-11] + "converted_img_"+d+"_"+str(i)+".png"
        img1 = cv2.imread(path2)
        #img2 = cv2.imread("/home/yara/drone_depth/Semantic_M4Depth/scripts/midair_conversion/000008_s.png")
        img2 = cv2.imread(path1)
        #img1 = cv2.resize(image1, [384,384], interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, [384,384], interpolation=cv2.INTER_LINEAR)
        
        image1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
        print(f"SSIM Score: ", round(ssim_score[0], 2))
        
        hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        
        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
        print(f"Histogram Similarity Score: ", round(metric_val, 2))
        
        mse_value = mse(image1_gray, image2_gray)
        print(f"MSE Value: ",round(mse_value,2))
'''


