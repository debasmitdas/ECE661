# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 19:14:19 2016

@author: Debasmit Das
"""
import cv2
import numpy as np
import math

img1=cv2.imread('flatiron.jpg')
img2=cv2.imread('monalisa')
img3=cv2.imread('wideangle.jpg')


#Parameters of Image 1
pt1=np.array([189.75,122.3,1])
pt2=np.array([320.6,81.5,1])
pt3=np.array([150.9,256.6,1])
#pt4=np.array([316,224,1])
pt4=np.array([326,177,1])
pt5=np.array([309.2,256.6,1])
#pt6=np.array([426,204,1])
pt6=np.array([432,154,1])
pt7=np.array([420,241.75,1])
pt8=np.array([227.45,383.45,1])
pt9=np.array([192.6,532.6,1])
pt10=np.array([330.9,377.15,1])
pt11=np.array([297,543,1])

l1=np.cross(pt1,pt2)
m1=np.cross(pt1,pt3)
l2=np.cross(pt4,pt6)
m2=np.cross(pt4,pt5)
l3=np.cross(pt6,pt7)
m3=np.cross(pt5,pt7)
l4=np.cross(pt8,pt10)
m4=np.cross(pt8,pt9)
l5=np.cross(pt10,pt11)
m5=np.cross(pt11,pt9)

#Normalizing the lines
l1=l1/l1[2];
m1=m1/m1[2];
l2=l2/l2[2];
m2=m2/m2[2];
l3=l3/l3[2];
m3=m3/m3[2];
l4=l4/l4[2];
m4=m4/m4[2];
l5=l5/l5[2];
m5=m5/m5[2];



paramM=np.zeros((5,5))
paramM[0]=np.array([l1[0]*m1[0], 0.5*(l1[1]*m1[0]+l1[0]*m1[1]), l1[1]*m1[1], 0.5*(l1[0]*m1[2]+l1[2]*m1[0]), 0.5*(l1[2]*m1[1]+l1[1]*m1[2])])
paramM[1]=np.array([l2[0]*m2[0], 0.5*(l2[1]*m2[0]+l2[0]*m2[1]), l2[1]*m2[1], 0.5*(l2[0]*m2[2]+l2[2]*m2[0]), 0.5*(l2[2]*m2[1]+l2[1]*m2[2])])
paramM[2]=np.array([l3[0]*m3[0], 0.5*(l3[1]*m3[0]+l3[0]*m3[1]), l3[1]*m3[1], 0.5*(l3[0]*m3[2]+l3[2]*m3[0]), 0.5*(l3[2]*m3[1]+l3[1]*m3[2])])
paramM[3]=np.array([l4[0]*m4[0], 0.5*(l4[1]*m4[0]+l4[0]*m4[1]), l4[1]*m4[1], 0.5*(l4[0]*m4[2]+l4[2]*m4[0]), 0.5*(l4[2]*m4[1]+l4[1]*m4[2])])
paramM[4]=np.array([l5[0]*m5[0], 0.5*(l5[1]*m5[0]+l5[0]*m5[1]), l5[1]*m5[1], 0.5*(l5[0]*m5[2]+l5[2]*m5[0]), 0.5*(l5[2]*m5[1]+l5[1]*m5[2])])

paramV=np.zeros((5,1))
paramV[0]=-l1[2]*m1[2];
paramV[1]=-l2[2]*m2[2];
paramV[2]=-l3[2]*m3[2];
paramV[3]=-l4[2]*m4[2];
paramV[4]=-l5[2]*m5[2];

paramA=np.zeros((5,1))
paramA=np.dot(np.linalg.inv(paramM),paramV)

S=np.zeros((2,2))
S[0,0]=paramA[0]
S[0,1]=0.5*paramA[1]
S[1,0]=0.5*paramA[1]
S[1,1]=paramA[2]

U,D,V=np.linalg.svd(S)

K=np.zeros((2,2))
D_K=np.diag(np.sqrt(D))
K=np.dot(U,np.dot(D_K,U))

v=np.zeros((2,1))
vec=np.zeros((2,1))
vec[0]=0.5*paramA[3]
vec[1]=0.5*paramA[4]

v=np.dot(np.linalg.inv(K),vec)

H=np.zeros((3,3))
H[:,2]=np.array([0,0,1]).T
H[:,1]=np.array([K[0,1],K[1,1],v[1]]).T
H[:,0]=np.array([K[0,0],K[1,0],v[0]]).T

#parameters of Image 1
bnd1=np.dot(np.linalg.inv(H),np.array([0,0,1]).T)
bnd2=np.dot(np.linalg.inv(H),np.array([0,img1.shape[1],1]).T)
bnd3=np.dot(np.linalg.inv(H),np.array([img1.shape[0],0,1]).T)
bnd4=np.dot(np.linalg.inv(H),np.array([img1.shape[0],img1.shape[1],1]).T)