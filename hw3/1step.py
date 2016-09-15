# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 19:14:19 2016

@author: Debasmit Das
"""
import cv2
import numpy as np
import math

img1=cv2.imread('flatiron.jpg')



def dim_offset(Hom,image):
	Points=np.array([[0,0,1],[0,image.shape[1],1],[image.shape[0],0,1],[image.shape[0],image.shape[1],1]])
	tmp=np.zeros((Points.shape[1],Points.shape[0]))
	tmp=np.array((np.dot(Hom,Points.T)).T)
	for i in range(0,Points.shape[0]):
		tmp[i]=tmp[i]/tmp[i,2]
	tmp=tmp.T ;
      
	# Getting Offsets and New Dimensions
	offset_X=round(min(tmp[0]))
	offset_Y=round(min(tmp[1]))
	dim_X=(max(tmp[0])-offset_X)
	dim_Y=(max(tmp[1])-offset_Y)	
	return offset_X,offset_Y,dim_X,dim_Y


# Getting RGB data function using weighted average of distance from corner points starts here
def getdata(point, img):
    tp_left =img[(math.floor(point[0])),(math.floor(point[1]))]
    tp_right =img[math.floor(point[0]),math.floor(point[1]+1)]
    bt_left =img[math.floor(point[0]+1),math.floor(point[1])]
    bt_right =img[math.floor(point[0]+1),math.floor(point[1]+1)]
    diff_x = point[0] - math.floor(point[0])
    diff_y = point[1] - math.floor(point[1])
    tp_left_weight= pow(pow(diff_x,2)+pow(diff_y,2),-0.5)
    tp_right_weight = pow(pow(diff_x,2)+pow(1-diff_y,2),-0.5)
    bt_left_weight = pow(pow(1-diff_x,2)+pow(diff_y,2),-0.5)
    bt_right_weight = pow(pow(1-diff_x,2)+pow(1-diff_y,2),-0.5)
    result_pt = (tp_left*tp_left_weight+tp_right*tp_right_weight+bt_left*bt_left_weight+bt_right*bt_right_weight)/(tp_left_weight+tp_right_weight+bt_left_weight+bt_right_weight)
    return result_pt
# getting RGB data function using weighted average ends here
   
# Image mapping code starts here
def img_map(dest_img,Hom,Off_X,Off_Y,Dim_X,Dim_Y):
	# creating aspect ratio
	Aspect_ratio=float(Dim_Y)/float(Dim_X)
	if Dim_Y<Dim_X:	
		Dim_World_Y=dest_img.shape[1]
		Dim_World_X=math.ceil(Dim_World_Y/Aspect_ratio)
	else:
		Dim_World_X=dest_img.shape[0]
		Dim_World_Y=math.ceil(Dim_World_X*Aspect_ratio)
	# Determining Scaling factor 
	p_ht=Dim_X/Dim_World_X
	p_wd=Dim_Y/Dim_World_Y
	
	tmp_img=np.zeros((int(Dim_World_X),int(Dim_World_Y),3),dtype='uint8')
	invHom=np.linalg.inv(Hom)
	for i in range(0,int(Dim_World_X)):
		for j in range(0,int(Dim_World_Y)):
			point_tmp = np.array([i*p_ht, j*p_wd, 1])
			point_new = point_tmp+np.array([Off_X, Off_Y, 0])
			trans_coord = np.array(np.dot(invHom,point_new))
			trans_coord = trans_coord/trans_coord[2]
			if (trans_coord[0]>0) and (trans_coord[0]<dest_img.shape[0]-1) and (trans_coord[1]>0) and (trans_coord[1]<dest_img.shape[1]-1):
				tmp_img[i][j]=getdata(trans_coord,dest_img)
	return tmp_img

# These the are final points used for flatiron image
pt1=np.array([191,122,1],dtype='float')
pt3=np.array([492,29,1],dtype='float')
pt2=np.array([65,541,1],dtype='float')
pt4=np.array([408,586,1],dtype='float')

l1=np.cross(pt1,pt3)
m1=np.cross(pt1,pt2)
l2=np.cross(pt1,pt2)
m2=np.cross(pt2,pt4)
l3=np.cross(pt3,pt4)
m3=np.cross(pt2,pt4)
l4=np.cross(pt3,pt4)
m4=np.cross(pt1,pt3)
l5=np.cross(pt1,pt4)
m5=np.cross(pt2,pt3)





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



paramM=np.zeros((5,5),dtype='float')
paramM[0]=np.array([l1[0]*m1[0], 0.5*(l1[1]*m1[0]+l1[0]*m1[1]), l1[1]*m1[1], 0.5*(l1[0]*m1[2]+l1[2]*m1[0]), 0.5*(l1[2]*m1[1]+l1[1]*m1[2])])
paramM[1]=np.array([l2[0]*m2[0], 0.5*(l2[1]*m2[0]+l2[0]*m2[1]), l2[1]*m2[1], 0.5*(l2[0]*m2[2]+l2[2]*m2[0]), 0.5*(l2[2]*m2[1]+l2[1]*m2[2])])
paramM[2]=np.array([l3[0]*m3[0], 0.5*(l3[1]*m3[0]+l3[0]*m3[1]), l3[1]*m3[1], 0.5*(l3[0]*m3[2]+l3[2]*m3[0]), 0.5*(l3[2]*m3[1]+l3[1]*m3[2])])
paramM[3]=np.array([l4[0]*m4[0], 0.5*(l4[1]*m4[0]+l4[0]*m4[1]), l4[1]*m4[1], 0.5*(l4[0]*m4[2]+l4[2]*m4[0]), 0.5*(l4[2]*m4[1]+l4[1]*m4[2])])
paramM[4]=np.array([l5[0]*m5[0], 0.5*(l5[1]*m5[0]+l5[0]*m5[1]), l5[1]*m5[1],  0.5*(l5[0]*m5[2]+l5[2]*m5[0]), 0.5*(l5[2]*m5[1]+l5[1]*m5[2])])


paramV=np.zeros((5,1))
paramV[0]=-l1[2]*m1[2];
paramV[1]=-l2[2]*m2[2];
paramV[2]=-l3[2]*m3[2];
paramV[3]=-l4[2]*m4[2];
paramV[4]=-l5[2]*m5[2];

# paramA solves for a,b,c,d,e,f=1
paramA=np.zeros((5,1))
paramA=np.dot(np.linalg.pinv(paramM),paramV)

S=np.zeros((2,2))
S[0,0]=paramA[0]
S[0,1]=0.5*paramA[1]
S[1,0]=0.5*paramA[1]
S[1,1]=paramA[2]

#Singular value decompostion of S
U,D,V=np.linalg.svd(S)

K=np.zeros((2,2))
D_K=np.diag(np.sqrt(D))
K=np.dot(V,np.dot(D_K,V))




v=np.zeros((2,1))
vec=np.zeros((2,1))
vec[0]=0.5*paramA[3]
vec[1]=0.5*paramA[4]

v=np.dot(np.linalg.pinv(K),vec)

H=np.zeros((3,3))

#Filling up the homography
H[0]=np.array([K[0,0],K[0,1],0])
H[1]=np.array([K[1,0],K[1,1],0])
H[2]=np.array([v[0],v[1],1])

invH=np.linalg.inv(H)

# Dimension offset is carried out
[Off_X,Off_Y,Dim_X,Dim_Y]= dim_offset(invH,img1)
# Image Mapping is carried out
output_persp=img_map(img1,invH,Off_X,Off_Y,Dim_X,Dim_Y)
cv2.imwrite('Result.jpg',output_persp)
