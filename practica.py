import numpy as np
from matplotlib import pyplot as plt
import cv2

ìndex_training_ini = 2
index_testing = 33
index_training = 48
#500 descriptores se generan
print("hola")

for x in range(ìndex_training_ini+1):

    img = cv2.imread("training/frontal_"+str(x+1)+".jpg",0)
    # Initiate ORB detector
    orb = cv2.ORB_create((int nfeatures=3, float scaleFactor=1.0f, int nlevels=1)
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    for y in des:
        print(des[y])
    img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2),plt.show()
# draw only keypoints location,not size and orientation
