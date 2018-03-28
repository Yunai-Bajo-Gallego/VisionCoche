import numpy as np
from matplotlib import pyplot as plt
import cv2

index_testing = 33
index_training = 48
#500 descriptores se generan

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
"""
img1 = cv2.imread("training/frontal_1.jpg",0) # queryImage
img2 = cv2.imread("training/frontal_2.jpg",0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

for k in kp1:
    print(k.pt[0],"-",k.pt[1])

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params) #método add

flann.add([des1])

#Guardar en lista los keypoints. recorrerlos y tener una sola lista con los keypoints.

#flann.add(des2)

#matches = flann.knnMatch(queryDescriptors=des1,trainDescriptors=des2,k=2)

matches = flann.knnMatch(des2,k=2)
j=0
i=0
for r in matches:
    print(" ",j," ",r)
    j=j+1
    for m in r:                                                        #de lso que se parecen guardamos este index. quryIndex    
        print(i," ","Resultado - distancia:",m.distance," img: ",m.imgIdx," queryIdx: ", m.queryIdx," trainIdx:",m.trainIdx)
        i=i+1
#guardar en res. res = []
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        
draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)
img3 = cv2.drawMatchesKnn(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2,matches1to2=matches,outImg=None,**draw_params)
plt.imshow(img3,),plt.show()
"""









# Creamos SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Creamos FLANN 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params) #método add

# Creamos array de keypoints
lista_kp = []
# Creamos array de vectores
lista_vectores = []

#Recorremos imagenes de entrenamiento y añadimos sus descriptores al FLANN
for x in range (1,index_training):
    img = cv2.imread("training/frontal_"+str(x)+".jpg",0)
    # Calculamos centro de la imagen para sacar el vector (En este caso siempre es el mismo pero por si cambiase) 
    centroXimg = img.shape[1]/2
    centroYimg = img.shape[0]/2
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img,None)    
    for k in kp:
        #Añadimos los kp a una lista
        lista_kp.append(k)
        #Añadimos el vector del keypoint hacia el centro en una lista
        vector = [centroXimg-k.pt[0],centroYimg-k.pt[1]]
        lista_vectores.append(vector)
        
    flann.add([des])

#Cargamos la imagen del test
#imgTest = cv2.imread("testing/test1.jpg",0)
imgTest = cv2.imread("training/frontal_22.jpg",0)

kp, des = sift.detectAndCompute(imgTest,None)
#Hacemos el match y obtenemos coincidencias
matches = flann.knnMatch(des,k=2)

j = 0
i = 0
print(len(matches)," ",len(lista_kp)," ",len(lista_vectores)," ",len(kp)," ",len(des))

#Creamos matriz de votación
sizeX = int(imgTest.shape[1]/10)
sizeY = int(imgTest.shape[0]/10)
print("Tamaño matriz: x=",sizeX," y=",sizeY)
matriz_votacion = [[0 for x in range(sizeY+1)] for y in range(sizeX+1)] #OJO

coorXVotacion = 0
coorYvotacion = 0

for r in matches:
    #print(" ",j," ",r)
    j=j+1
    for m in r:                                                        
        #print(i," ","Resultado - distancia:",m.distance," img: ",m.imgIdx," queryIdx: ", m.queryIdx," trainIdx:",m.trainIdx)
        #Para votar:  coorXdelKp (+ ó *) (vector[m.trainIdx]*kp[queryIdx].size) / lista_kp[trainIdx].size
        indexX = kp[m.queryIdx].pt[0] + ((lista_vectores[m.trainIdx][0]*kp[m.queryIdx].size) / lista_kp[m.trainIdx].size)
        indexX = int(indexX / 10)
        indexY = kp[m.queryIdx].pt[1] + ((lista_vectores[m.trainIdx][1]*kp[m.queryIdx].size) / lista_kp[m.trainIdx].size)
        indexY = int(indexY / 10)
        print(indexX,"-",indexY)
        if indexX>=0 and indexY>=0 and indexX<sizeX and indexY<sizeY:
            matriz_votacion[indexX][indexY] = matriz_votacion[indexX][indexY] + 1      
            if(matriz_votacion[indexX][indexY] > matriz_votacion[coorXVotacion][coorYvotacion]):
                coorXVotacion = indexX
                coorYvotacion = indexY
        i=i+1


print("MATRIZ: ")
print(matriz_votacion)
print("coordenadas votadas: x=",coorXVotacion," y=",coorYvotacion)

imgFinal = cv2.drawKeypoints(imgTest,kp,None,color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow(imgFinal),plt.show()


imgFinal = cv2.circle(imgFinal, (coorXVotacion*10, coorYvotacion*10), 5, (255,0,0), 2)
plt.imshow(imgFinal)
plt.show()
        
