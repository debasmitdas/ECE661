import cv2
import numpy as np
import math


img1 = cv2.imread('flatiron.jpg')
# Finding homography to remove perspective distortion
# Function to find homography using 2 pair of parallel lines
def per_hom_lines(points):
	""" As per notes Finding the cross product of the two points to find the line connecting the points
	l1 is formed by the cross product of P,Q. l1 is formed by the cross product of P,Q. l3 is formed by P,R and l4 is formed by Q,S """
	l1= np.cross((points[2,:]),(points[3,:]))
 	l2= np.cross(points[0,:],points[1,:])
	l3= np.cross(points[2,:],points[0,:])
	l4= np.cross(points[3,:],points[1,:])
	# Normalizing the lines
	l1_n= l1/l1[2]
	l2_n= l2/l2[2]
	l3_n= l3/l3[2]
	l4_n= l4/l4[2]
	"""Finding the two ideal points by the cross product of the lines obtained A = cross product l1 and l2. while B= crossproduct of l3 and l4."""
	A= np.cross(l3_n,l4_n)
	B= np.cross(l1_n,l2_n)
	# Normalizing the points
	A_n= A/A[2]
	B_n= B/B[2]
	""" Finding the line at infinity. Normalizing the line at infinity"""
	l_inf= np.cross(A_n,B_n) 	
 	l_inf_n= l_inf/l_inf[2]
	# Generating homography matrix
	H = np.matrix([[1.0,0,0],[0,1.0,0],[0,0,0]])
	H[2] = l_inf_n
	return H
 
# Function to find homography using 2 pair of parallel lines ends here

# Finding the dimensions and offset for the new image in world plane
def dim_offset(Hom,image):
	Points=np.array([[0,0,1],[0,image.shape[1],1],[image.shape[0],0,1],[image.shape[0],image.shape[1],1]])
	tmp=np.zeros((Points.shape[1],Points.shape[0]))
	tmp=np.array((np.dot(Hom,Points.T)).T)
	for i in range(0,Points.shape[0]):
		tmp[i]=tmp[i]/tmp[i,2]
	tmp=tmp.T
	# Generating Offsets and New Dimensions
	offset_X=round(min(tmp[0]))
	offset_Y=round(min(tmp[1]))
	dim_X=(max(tmp[0])-offset_X)
	dim_Y=(max(tmp[1])-offset_Y)	
	return offset_X,offset_Y,dim_X,dim_Y


# Getting RGB data function using weighted average starts here
def getdata(point, img):
    tp_left =img[(math.floor(point[0,0])),(math.floor(point[0,1]))]
    tp_right =img[math.floor(point[0,0]),math.floor(point[0,1]+1)]
    bt_left =img[math.floor(point[0,0]+1),math.floor(point[0,1])]
    bt_right =img[math.floor(point[0,0]+1),math.floor(point[0,1]+1)]
    diff_x = point[0,0] - math.floor(point[0,0])
    diff_y = point[0,1] - math.floor(point[0,1])
    tp_left_weight= pow(pow(diff_x,2)+pow(diff_y,2),-0.5)
    tp_right_weight = pow(pow(diff_x,2)+pow(1-diff_y,2),-0.5)
    bt_left_weight = pow(pow(1-diff_x,2)+pow(diff_y,2),-0.5)
    bt_right_weight = pow(pow(1-diff_x,2)+pow(1-diff_y,2),-0.5)
    result_pt = (tp_left*tp_left_weight+tp_right*tp_right_weight+bt_left*bt_left_weight+bt_right*bt_right_weight)/(tp_left_weight+tp_right_weight+bt_left_weight+bt_right_weight)
    return result_pt

    
# Image mapping code 
def image_map(dest_image,Homo,Off_X,Off_Y,Dim_X,Dim_Y):
	# creating aspect ratio 
	Aspect_ratio=float(Dim_Y)/float(Dim_X)
	if Dim_Y<Dim_X:	
		Dim_World_Y=dest_image.shape[1]
		Dim_World_X=math.ceil(Dim_World_Y/Aspect_ratio)
	else:
		Dim_World_X=dest_image.shape[0]
		Dim_World_Y=math.ceil(Dim_World_X*Aspect_ratio)
	# Determining Scaling factor 
	p_ht=Dim_X/Dim_World_X
	p_wd=Dim_Y/Dim_World_Y
	# creating a base image for the projecting the corrected image
	tmpimg=np.zeros((int(Dim_World_X),int(Dim_World_Y),3),dtype='uint8')
	invHom=np.linalg.inv(Homo)
	for i in range(0,int(Dim_World_X)):
		for j in range(0,int(Dim_World_Y)):
			point_tmp = np.array([i*p_ht, j*p_wd, 1])
			point_new = point_tmp+np.array([Off_X, Off_Y, 0])
			trans_coord = np.array(np.dot(invHom,point_new))
			trans_coord = trans_coord/trans_coord[0,2]
			if (trans_coord[0,0]>0) and (trans_coord[0,0]<dest_image.shape[0]-1) and (trans_coord[0,1]>0) and (trans_coord[0,1]<dest_image.shape[1]-1):
				tmpimg[i][j]=getdata(trans_coord,dest_image)
	return tmpimg


# Finding Homography of Affine
def Homography_Affine(points):
	# Finding two orthogonal lines from 3 points
	L1=np.cross(points[1,:],points[0,:])
	M1=np.cross(points[1,:],points[2,:])
	L2=np.cross(points[4,:],points[3,:])
	M2=np.cross(points[4,:],points[5,:])
	# Find S matrix from linear quation s=(A^-1)*b
	Mat_A=np.zeros((2,2),dtype='float')
	b=np.array([0,0])
	Mat_A[0]=np.array([L1[0]*M1[0],M1[0]*L1[1]+M1[1]*L1[0]])
	Mat_A[1]=np.array([L2[0]*M2[0],M2[0]*L2[1]+M2[1]*L2[0]])
	b=np.array([-M1[1]*L1[1],-M2[1]*L2[1]])
	[s11, s12]= (np.linalg.pinv(Mat_A).dot(b))
	print s11,s12
	S=np.zeros((2,2),dtype='float')
	# Generating S matrix which is ATA
	S[0]=np.array([s11,s12])
	S[1]=np.array([s12,1])
	# Find SVD decomposition of S to find A
	V,D_S,Vt=np.linalg.svd(S,full_matrices=1)
	D_A=np.sqrt(D_S)
	# Reconstructing D_A matrix from the S
	D = np.zeros((2,2),dtype='float')
	D[0]= np.array([D_A[0],0])
	D[1]= np.array([0,D_A[1]])
	A=V.dot(D).dot(Vt)
	# Finally the homography is 
	H=np.matrix([[A[0,0],A[0,1],0],[A[1,0],A[1,1],0],[0,0,1]])
	return H

Points_1=np.array([[274,96,1],[153,552,1],[493,28,1],[412,581,1]],dtype='float')
H_proj = per_hom_lines(Points_1)
[Off_X,Off_Y,Dim_X,Dim_Y]= dim_offset(H_proj,img1)
output_proj=image_map(img1,H_proj,Off_X,Off_Y,Dim_X,Dim_Y)
cv2.imwrite('Result1.jpg',output_proj)
input_aff_1=output_proj
# Points to find pair of line that will be orthogonal in undistorted image
points_affine_1=np.array([[346,231,1],[314,248,1],[301,278,1],[275,338,1],[263,365,1],[294,349,1]])
H_Affine=Homography_Affine(points_affine_1)
H_Affine_inv=np.linalg.inv(H_Affine)
[Off_XA,Off_YA,Dim_XA,Dim_YA] = dim_offset(H_Affine_inv,input_aff_1)
output_affine=image_map(input_aff_1,H_Affine_inv,Off_XA,Off_YA,Dim_XA,Dim_YA)
cv2.imwrite('Result2.jpg',output_affine)

####################################################

