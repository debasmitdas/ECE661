
#Written  by Debasmit Das

#Useful modules are loaded
import numpy as np
import cv2
import math

#The image files are read
img1=cv2.imread('1.jpg')
img2=cv2.imread('2.jpg')
img3=cv2.imread('3.jpg')

#The image file output is initialised
imgnew=np.zeros((img3.shape[0],img3.shape[1],3))

#function to get color values
def getrgb(pt, img):
    p = img[math.floor(pt[0,0]), math.floor(pt[0,1])]
    q = img[math.floor(pt[0,0]), math.ceil(pt[0,1])]
    r = img[math.ceil(pt[0,0]), math.floor(pt[0,1])]
    s = img[math.ceil(pt[0,0]), math.ceil(pt[0,1])]

    x = pt[0,0] - math.floor(pt[0,0])
    y = pt[0,1] - math.floor(pt[0,1])

    wp = 1/np.linalg.norm(np.array([x,y]))
    wq = 1/np.linalg.norm(np.array([x,1-y]))
    wr = 1/np.linalg.norm(np.array([1-x,y]))
    ws = 1/np.linalg.norm(np.array([1-x,1-y]))

    color = (p*wp+q*wq+r*wr+s*ws)/(wp+wq+wr+ws)

    return color
## rgb values of the pixels are found

#We have to interchange the x and y co-ordinates in implementation
#This is because of way images are stored in tensors
#PQRS co-ordinates of all the images are found
p1=np.matrix('430 2100 1', dtype=float)
q1=np.matrix('540 3300 1', dtype=float)
r1=np.matrix('1490 2150 1', dtype=float)
s1=np.matrix('1350 3310 1', dtype=float)
p2=np.matrix('800 1620 1', dtype=float)
q2=np.matrix('770 2980 1', dtype=float)
r2=np.matrix('1600 1620 1', dtype=float)
s2=np.matrix('1520 2990 1', dtype=float)
p3=np.matrix('560 1000 1', dtype=float)
q3=np.matrix('420 2400 1', dtype=float)
r3=np.matrix('1410 1030 1', dtype=float)
s3=np.matrix('1480 2400 1', dtype=float)

# H1 homography values
P12=np.matrix(np.zeros((8,8)), dtype=float)

# H2 homography values
P23=np.matrix(np.zeros((8,8)), dtype=float)

#Declaring matrix values for mapping between

P12[0,0:3] = p2; P23[0,0:3] = p3
P12[0,6:9] = -p2[0,0:2]*p1[0,0]; P23[0,6:9] = -p3[0,0:2]*p2[0,0]
P12[1,3:6] = p2; P23[1,3:6] = p3;
P12[1,6:9] = -p2[0,0:2]*p1[0,1] ; P23[1,6:9] = -p3[0,0:2]*p2[0,1]
P12[2,0:3] = q2 ; P23[2,0:3] = q3
P12[2,6:9] = -q2[0,0:2]*q1[0,0]; P23[2,6:9] = -q3[0,0:2]*q2[0,0]
P12[3,3:6] = q2 ; P23[3,3:6] = q3
P12[3,6:9] = -q2[0,0:2]*q1[0,1] ; P23[3,6:9] = -q3[0,0:2]*q2[0,1]
P12[4,0:3] = r2 ; P23[4,0:3] = r3
P12[4,6:9] = -r2[0,0:2]*r1[0,0] ; P23[4,6:9] = -r3[0,0:2]*r2[0,0]
P12[5,3:6] = r2 ; P23[5,3:6] = r3
P12[5,6:9] = -r2[0,0:2]*r1[0,1] ; P23[5,6:9] = -r3[0,0:2]*r2[0,1]
P12[6,0:3] = s2 ; P23[6,0:3] = s3
P12[6,6:9] = -s2[0,0:2]*s1[0,0] ; P23[6,6:9] = -s3[0,0:2]*s2[0,0]
P12[7,3:6] = s2 ; P23[7,3:6] = s3
P12[7,6:9] = -s2[0,0:2]*s1[0,1] ; P23[7,6:9] = -s3[0,0:2]*s2[0,1]



t12 = np.matrix('0 0 0 0 0 0 0 0',dtype=float) ; t23 = np.matrix('0 0 0 0 0 0 0 0',dtype=float)
t12[0,0:2] = p1[0,0:2] ; t23[0,0:2] = p1[0,0:2]
t12[0,2:4] = q1[0,0:2] ; t23[0,2:4] = q1[0,0:2]
t12[0,4:6] = r1[0,0:2] ; t23[0,4:6] = r1[0,0:2]
t12[0,6:8] = s1[0,0:2] ; t23[0,6:8] = s1[0,0:2]

#Finding the Homography
param1 = np.transpose(np.linalg.inv(P12)*np.transpose(t12)) ; param2 = np.transpose(np.linalg.inv(P23)*np.transpose(t23))
H1 = np.zeros((3,3),dtype=float) ; H2 = np.zeros((3,3),dtype=float)
H1[0] = param1[0,0:3] ; H2[0] = param2[0,0:3]
H1[1] = param1[0,3:6] ; H2[1] = param2[0,3:6]
H1[2,0:2] = param1[0,6:8] ; H2[2,0:2] = param2[0,6:8]
H1[2,2] = 1 ; H2[2,2] = 1


#The composite homogeneous matrix is found
H=np.dot(H1,H2)


#Looping over the image
tmp = np.matrix('0 0 1',dtype=float)
for row in range(0,imgnew.shape[0]-1):
    for col in range(0,imgnew.shape[1]-1):
        #if img1fill[row,column,1] > 0:
            tmp[0,0] = row
            tmp[0,1] = col
            tr = np.transpose(H*np.transpose(tmp))
            tr = tr/tr[0,2]
            if tr[0,0]>0 and tr[0,1]>0 and tr[0,0]<imgnew.shape[0]-1 and tr[0,1]<imgnew.shape[1]-1 :
                imgnew[row,col]=getrgb(tr,img1)



cv2.imwrite('Result.jpg',imgnew)
cv2.destroyAllWindows()


