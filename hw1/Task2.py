
#Written  by Debasmit Das

import numpy as np
import cv2
import math

img1=cv2.imread('1.jpg')
img2=cv2.imread('2.jpg')


img3=cv2.imread('3.jpg')
imgs=cv2.imread('Seinfeld.jpg')

img1fill= np.zeros((img3.shape[0],img3.shape[1],3),dtype='uint8')
imgnew=np.zeros((img3.shape[0],img3.shape[1],3),dtype='uint8')

#pts = np.array([[2100,430],[3300,540],[3310,1350],[2150,1490]],np.int32)
#pts = np.array([[1620,800],[2980,770],[2990,1520],[1620,1600]],np.int32)
#pts = np.array([[1000,560],[2400,420],[2400,1480],[1030,1410]],np.int32)
#pts = np.array([[0,0],[4128,0],[4128,3096],[0,3096]],np.int32)
#pts = pts.reshape((-1,1,2))
#cv2.fillPoly(img1fill,[pts],(0.0.))
#cv2.imshow('image',img1fill)
#cv2.waitKey(0)
cv2.imwrite('selectRegion.jpg',img1fill)
#cv2.destroyAllWindows()

#My Points
#The order is P, Q, R, S

#My Point's

#This is the original pixel matrix
"""
Points_1=array([[2100,430,1],[3300,540,1],[2150,1490,1],[3310,1350,1]])
Points_2=array([[1620,800,1],[2980,770,1],[1620,1600,1],[2990,1520,1]])
Points_3=array([[1000,560,1],[2400,420,1],[1030,1410,1],[2400,1480,1]])
Points_s=array([[0,0,1],[2560,0,1],[0,1536,1],[2560,1536,1]])
"""

def getcolor(point, img):
    p = img[math.floor(point[0,0])%img.shape[0],math.floor(point[0,1])%img.shape[1]]
    q = img[math.floor(point[0,0])%img.shape[0],math.floor(point[0,1]+1)%img.shape[1]]
    r =img[math.floor(point[0,0]+1)%img.shape[0],math.floor(point[0,1])%img.shape[1]]
    s = img[math.floor(point[0,0]+1)%img.shape[0],math.floor(point[0,1]+1)%img.shape[1]]

    x = point[0,0] - math.floor(point[0,0])
    y = point[0,1] - math.floor(point[0,1])

    pweight= pow(pow(x,2)+pow(y,2),-0.5)
    qweight = pow(pow(x,2)+pow(1-y,2),-0.5)
    rweight = pow(pow(1-x,2)+pow(y,2),-0.5)
    sweight = pow(pow(1-x,2)+pow(1-y,2),-0.5)
    newpoint = (p*pweight+q*qweight+r*rweight+s*sweight)/(pweight+qweight+rweight+sweight)

    return newpoint
## getcolor function end



#We have to interchange the x and y co-ordinates

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

ps=np.matrix('0 0 1', dtype=float)
qs=np.matrix('0 2560 1', dtype=float)
rs=np.matrix('1536 0 1', dtype=float)
ss=np.matrix('1536 2560 1', dtype=float)

#Initializing parameter Matrix for seinfield image to 1
paraMs1=np.zeros((8,8), dtype=float)
paraMs1=np.matrix(paraMs1, dtype=float)

#Initializing parameter Matrix for seinfield image to 2
paraMs2=np.zeros((8,8), dtype=float)
paraMs2=np.matrix(paraMs1, dtype=float)

#Initializing parameter Matrix for seinfield image to 3
paraMs3=np.zeros((8,8), dtype=float)
paraMs3=np.matrix(paraMs1, dtype=float)

paraM12=np.zeros((8,8), dtype=float)
paraM12=np.matrix(paraM12, dtype=float)


paraM23=np.zeros((8,8), dtype=float)
paraM23=np.matrix(paraM23, dtype=float)

#Declaring matrix values for mapping between s and 1
"""
paraMs1[0,0:3] = p1
paraMs1[0,6:9] = -p1[0,0:2]*ps[0,0]
paraMs1[1,3:6] = p1
paraMs1[1,6:9] = -p1[0,0:2]*ps[0,1]
paraMs1[2,0:3] = q1
paraMs1[2,6:9] = -q1[0,0:2]*qs[0,0]
paraMs1[3,3:6] = q1
paraMs1[3,6:9] = -q1[0,0:2]*qs[0,1]
paraMs1[4,0:3] = r1
paraMs1[4,6:9] = -r1[0,0:2]*rs[0,0]
paraMs1[5,3:6] = r1
paraMs1[5,6:9] = -r1[0,0:2]*rs[0,1]
paraMs1[6,0:3] = s1
paraMs1[6,6:9] = -s1[0,0:2]*ss[0,0]
paraMs1[7,3:6] = s1
paraMs1[7,6:9] = -s1[0,0:2]*ss[0,1]
"""

"""
paraMs2[0,0:3] = p2
paraMs2[0,6:9] = -p2[0,0:2]*ps[0,0]
paraMs2[1,3:6] = p2
paraMs2[1,6:9] = -p2[0,0:2]*ps[0,1]
paraMs2[2,0:3] = q2
paraMs2[2,6:9] = -q2[0,0:2]*qs[0,0]
paraMs2[3,3:6] = q2
paraMs2[3,6:9] = -q2[0,0:2]*qs[0,1]
paraMs2[4,0:3] = r2
paraMs2[4,6:9] = -r2[0,0:2]*rs[0,0]
paraMs2[5,3:6] = r2
paraMs2[5,6:9] = -r2[0,0:2]*rs[0,1]
paraMs2[6,0:3] = s2
paraMs2[6,6:9] = -s2[0,0:2]*ss[0,0]
paraMs2[7,3:6] = s2
paraMs2[7,6:9] = -s2[0,0:2]*ss[0,1]
"""

"""
paraMs3[0,0:3] = p3
paraMs3[0,6:9] = -p3[0,0:2]*ps[0,0]
paraMs3[1,3:6] = p3
paraMs3[1,6:9] = -p3[0,0:2]*ps[0,1]
paraMs3[2,0:3] = q3
paraMs3[2,6:9] = -q3[0,0:2]*qs[0,0]
paraMs3[3,3:6] = q3
paraMs3[3,6:9] = -q3[0,0:2]*qs[0,1]
paraMs3[4,0:3] = r3
paraMs3[4,6:9] = -r3[0,0:2]*rs[0,0]
paraMs3[5,3:6] = r3
paraMs3[5,6:9] = -r3[0,0:2]*rs[0,1]
paraMs3[6,0:3] = s3
paraMs3[6,6:9] = -s3[0,0:2]*ss[0,0]
paraMs3[7,3:6] = s3
paraMs3[7,6:9] = -s3[0,0:2]*ss[0,1]
"""

paraM12[0,0:3] = p2
paraM12[0,6:9] = -p2[0,0:2]*p1[0,0]
paraM12[1,3:6] = p2
paraM12[1,6:9] = -p2[0,0:2]*p1[0,1]
paraM12[2,0:3] = q2
paraM12[2,6:9] = -q2[0,0:2]*q1[0,0]
paraM12[3,3:6] = q2
paraM12[3,6:9] = -q2[0,0:2]*q1[0,1]
paraM12[4,0:3] = r2
paraM12[4,6:9] = -r2[0,0:2]*r1[0,0]
paraM12[5,3:6] = r2
paraM12[5,6:9] = -r2[0,0:2]*r1[0,1]
paraM12[6,0:3] = s2
paraM12[6,6:9] = -s2[0,0:2]*s1[0,0]
paraM12[7,3:6] = s2
paraM12[7,6:9] = -s2[0,0:2]*s1[0,1]



Rvector = np.matrix('0 0 0 0 0 0 0 0',dtype=float)
Rvector[0,0:2] = p1[0,0:2]
Rvector[0,2:4] = q1[0,0:2]
Rvector[0,4:6] = r1[0,0:2]
Rvector[0,6:8] = s1[0,0:2]

#parameter = paraMs1.I*Rvector.T
#parameter = paraMs1.T

parameter = paraM12.I*Rvector.T
parameter = parameter.T
H1 = np.zeros((3,3),dtype=float)
H1[0] = parameter[0,0:3]
H1[1] = parameter[0,3:6]
H1[2,0:2] = parameter[0,6:8]
H1[2,2] = 1

print H1

paraM23[0,0:3] = p3
paraM23[0,6:9] = -p3[0,0:2]*p2[0,0]
paraM23[1,3:6] = p3
paraM23[1,6:9] = -p3[0,0:2]*p2[0,1]
paraM23[2,0:3] = q3
paraM23[2,6:9] = -q3[0,0:2]*q2[0,0]
paraM23[3,3:6] = q3
paraM23[3,6:9] = -q3[0,0:2]*q2[0,1]
paraM23[4,0:3] = r3
paraM23[4,6:9] = -r3[0,0:2]*r2[0,0]
paraM23[5,3:6] = r3
paraM23[5,6:9] = -r3[0,0:2]*r2[0,1]
paraM23[6,0:3] = s3
paraM23[6,6:9] = -s3[0,0:2]*s2[0,0]
paraM23[7,3:6] = s3
paraM23[7,6:9] = -s3[0,0:2]*s2[0,1]



#Rvector = np.matrix('0 0 0 0 0 0 0 0',dtype=float)
Rvector[0,0:2] = p2[0,0:2]
Rvector[0,2:4] = q2[0,0:2]
Rvector[0,4:6] = r2[0,0:2]
Rvector[0,6:8] = s2[0,0:2]

#parameter = paraMs1.I*Rvector.T
#parameter = paraMs1.T

parameter = paraM23.I*Rvector.T
parameter = parameter.T
H2 = np.zeros((3,3),dtype=float)
H2[0] = parameter[0,0:3]
H2[1] = parameter[0,3:6]
H2[2,0:2] = parameter[0,6:8]
H2[2,2] = 1

print H2

H=np.dot(H1,H2)

temp = np.matrix('0 0 1',dtype=float)
for row in range(0,img3.shape[0]):
    for column in range(0,img3.shape[1]):
        #if img1fill[row,column,1] > 0:
            temp[0,0] = row
            temp[0,1] = column
            tr = H*temp.T
            tr = tr.T
            tr = tr/tr[0,2]
            if tr[0,0]>0 and tr[0,1]>0 and tr[0,0]<img3.shape[0] and tr[0,1]<img3.shape[1] :
                imgnew[row,column]=getcolor(tr,img1)


#img1[row,column]=img1[0,0]
#cv2.imshow('image',img1)
#cv2.waitKey(0)
cv2.imwrite('myNewImage.jpg',imgnew)
cv2.destroyAllWindows()


#Vishwa's Points
"""
Points_1a1=array([[421,2108,1],[531,3303,1],[1495,2148,1],[1357,3312,1]])
Points_1b=array([[795,1590,1],[768,2983,1],[1597,1616,1],[1518,2987,1]])
Points_1c=array([[562,999,1],[421,2425,1],[1412,1026,1],[1478,2406,1]])
Points_1d=array([[0,0,1],[0,2560,1],[1536,0,1],[1536,2560,1]])
"""