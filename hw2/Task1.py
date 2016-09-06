
#Written  by Debasmit Das

#Useful modules are loaded
import numpy as np
import cv2
import math

#Applying task 1 for the 3rd image
img3=cv2.imread('3.jpg')
imgs=cv2.imread('Seinfeld.jpg')

#Creating a box where the source image is filled
img1fill= np.zeros((img3.shape[0],img3.shape[1],3),dtype='uint8')
pts = np.array([[1000,560],[2400,420],[2400,1480],[1030,1410]],np.int32)
pts = pts.reshape((-1,1,2))
cv2.fillPoly(img1fill,[pts],(255,255,255))

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
p3=np.matrix('560 1000 1', dtype=float)
q3=np.matrix('420 2400 1', dtype=float)
r3=np.matrix('1410 1030 1', dtype=float)
s3=np.matrix('1480 2400 1', dtype=float)
ps=np.matrix('0 0 1', dtype=float)
qs=np.matrix('0 2560 1', dtype=float)
rs=np.matrix('1536 0 1', dtype=float)
ss=np.matrix('1536 2560 1', dtype=float)

Ps1=np.matrix(np.zeros((8,8)), dtype=float)

#Declaring matrix values for mapping between source and destination
Ps1[0,0:3] = p3
Ps1[0,6:9] = -p3[0,0:2]*ps[0,0]
Ps1[1,3:6] = p3
Ps1[1,6:9] = -p3[0,0:2]*ps[0,1]
Ps1[2,0:3] = q3
Ps1[2,6:9] = -q3[0,0:2]*qs[0,0]
Ps1[3,3:6] = q3
Ps1[3,6:9] = -q3[0,0:2]*qs[0,1]
Ps1[4,0:3] = r3
Ps1[4,6:9] = -r3[0,0:2]*rs[0,0]
Ps1[5,3:6] = r3
Ps1[5,6:9] = -r3[0,0:2]*rs[0,1]
Ps1[6,0:3] = s3
Ps1[6,6:9] = -s3[0,0:2]*ss[0,0]
Ps1[7,3:6] = s3
Ps1[7,6:9] = -s3[0,0:2]*ss[0,1]

ts1 = np.matrix('0 0 0 0 0 0 0 0',dtype=float)
ts1[0,0:2] = ps[0,0:2]
ts1[0,2:4] = qs[0,0:2]
ts1[0,4:6] = rs[0,0:2]
ts1[0,6:8] = ss[0,0:2]

#Finding the Homography
param = np.transpose(np.linalg.inv(Ps1)*np.transpose(ts1))
H = np.zeros((3,3),dtype=float)
H[0] = param[0,0:3]
H[1] = param[0,3:6]
H[2,0:2] = param[0,6:8]
H[2,2] = 1

#Looping over the image
tmp = np.matrix('0 0 1',dtype=float)
for row in range(0,img3.shape[0]-1):
    for col in range(0,img3.shape[1]-1):
        if img1fill[row,col,1] > 0:
            tmp[0,0] = row
            tmp[0,1] = col
            tr = np.transpose(H*np.transpose(tmp))
            tr = tr/tr[0,2]
            if tr[0, 0] > 0 and tr[0, 1] > 0 and tr[0, 0] < imgs.shape[0] - 1 and tr[0, 1] < imgs.shape[1] - 1:
                img3[row,col]=getrgb(tr,imgs)


cv2.imwrite('Result.jpg',img3)
cv2.destroyAllWindows()


