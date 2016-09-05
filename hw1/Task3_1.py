
#Written  by Debasmit Das

import numpy as np
import cv2
import math

img1=cv2.imread('1Task3.jpg')
img2=cv2.imread('2Task3.jpg')
img3=cv2.imread('3Task3.jpg')
imgs=cv2.imread('Debasmit.jpg')

img1fill= np.zeros((img1.shape[0],img1.shape[1],3),dtype='uint8')

pts = np.array([[788,190],[1090,213],[1055,770],[765,877]],np.int32)
#pts = np.array([[601,373],[952,381],[938,822],[593,833]],np.int32)
#pts = np.array([[536,157],[800,150],[784,778],[534,681]],np.int32)
pts = pts.reshape((-1,1,2))
cv2.fillPoly(img1fill,[pts],(255,255,255))
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

p1=np.matrix('190 788 1', dtype=float)
q1=np.matrix('213 1090 1', dtype=float)
r1=np.matrix('877 765 1', dtype=float)
s1=np.matrix('770 1055 1', dtype=float)

p2=np.matrix('373 601 1', dtype=float)
q2=np.matrix('381 952 1', dtype=float)
r2=np.matrix('833 593 1', dtype=float)
s2=np.matrix('822 938 1', dtype=float)

p3=np.matrix('157 536 1', dtype=float)
q3=np.matrix('150 800 1', dtype=float)
r3=np.matrix('681 534 1', dtype=float)
s3=np.matrix('778 784 1', dtype=float)

ps=np.matrix('0 0 1', dtype=float)
qs=np.matrix('0 438 1', dtype=float)
rs=np.matrix('484 0 1', dtype=float)
ss=np.matrix('484 438 1', dtype=float)

#Initializing parameter Matrix for seinfield image to 1
paraMs1=np.zeros((8,8), dtype=float)
paraMs1=np.matrix(paraMs1, dtype=float)

#Initializing parameter Matrix for seinfield image to 2
paraMs2=np.zeros((8,8), dtype=float)
paraMs2=np.matrix(paraMs1, dtype=float)

#Initializing parameter Matrix for seinfield image to 3
paraMs3=np.zeros((8,8), dtype=float)
paraMs3=np.matrix(paraMs1, dtype=float)

#Declaring matrix values for mapping between s and 1

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
Rvector = np.matrix('0 0 0 0 0 0 0 0',dtype=float)
Rvector[0,0:2] = ps[0,0:2]
Rvector[0,2:4] = qs[0,0:2]
Rvector[0,4:6] = rs[0,0:2]
Rvector[0,6:8] = ss[0,0:2]

#parameter = paraMs1.I*Rvector.T
#parameter = paraMs1.T

parameter = paraMs1.I*Rvector.T
parameter = parameter.T
H = np.zeros((3,3),dtype=float)
H[0] = parameter[0,0:3]
H[1] = parameter[0,3:6]
H[2,0:2] = parameter[0,6:8]
H[2,2] = 1

print H

temp = np.matrix('0 0 1',dtype=float)
for row in range(0,img1.shape[0]):
    for column in range(0,img1.shape[1]):
        if img1fill[row,column,1] > 0:
            temp[0,0] = row
            temp[0,1] = column
            tr = H*temp.T
            tr = tr.T
            tr = tr/tr[0,2]
            img1[row,column]=getcolor(tr,imgs)


#img1[row,column]=img1[0,0]
#cv2.imshow('image',img1)
#cv2.waitKey(0)
cv2.imwrite('myNewImage.jpg',img1)
cv2.destroyAllWindows()


#Vishwa's Points
"""
Points_1a1=array([[421,2108,1],[531,3303,1],[1495,2148,1],[1357,3312,1]])
Points_1b=array([[795,1590,1],[768,2983,1],[1597,1616,1],[1518,2987,1]])
Points_1c=array([[562,999,1],[421,2425,1],[1412,1026,1],[1478,2406,1]])
Points_1d=array([[0,0,1],[0,2560,1],[1536,0,1],[1536,2560,1]])
"""