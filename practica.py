import numpy as np
from matplotlib import pyplot as plt
import cv2

ìndex_training_ini = 2
index_testing = 33
index_training = 48
#500 descriptores se generan
print("hola")

#####################################################################################################
#for x in range(ìndex_training_ini+1):

    #img = cv2.imread("training/frontal_"+str(x+1)+".jpg",0)
    # Initiate ORB detector
    #orb = cv2.ORB_create(nfeatures=3,scaleFactor=1.0,nlevels=1)

    # find the keypoints with ORB
   # kp = orb.detect(img,None)
    # compute the descriptors with ORB
  #  kp, des = orb.compute(img, kp)

 #   img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#    plt.imshow(img2),plt.show()
# draw only keypoints location,not size and orientation
#########################################################################################################
"""
img1 = cv2.imread("training/frontal_1.jpg",0)          # queryImage
img2 = cv2.imread("training/frontal_3.jpg",0) # trainImage

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance) 

img3 = cv2.drawMatches(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2,matches1to2=matches[:10],outImg=None,flags=2)
plt.imshow(img3)
plt.show()
"""
##########################################################################################
"""
img1 = cv2.imread("training/frontal_1.jpg",0)# queryImage
img2 = cv2.imread("training/frontal_3.jpg",0)# trainImage
# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])     

# Draw first 10 matches.
##img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
img3 = cv2.drawMatchesKnn(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2,matches1to2=good,outImg=None,flags=2)

plt.imshow(img3),plt.show()

"""
################################################################################

img1 = cv2.imread("training/frontal_1.jpg",0)          # queryImage
img2 = cv2.imread("training/frontal_2.jpg",0) # trainImage
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params) #método add
matches = flann.knnMatch(queryDescriptors=des1,trainDescriptors=des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)
img3 = cv2.drawMatchesKnn(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2,matches1to2=matches,outImg=None,**draw_params)
plt.imshow(img3,),plt.show()
