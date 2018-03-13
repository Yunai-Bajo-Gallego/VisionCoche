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
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    variable = [kp[0],kp[1],kp[2]]
    print(variable[0])
    #for y in des:
        #print(des[y])
    img2 = cv2.drawKeypoints(img,variable,None,color=(0,255,0), flags=0)
    plt.imshow(img2),plt.show()
# draw only keypoints location,not size and orientation
