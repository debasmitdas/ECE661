"""
Author: Debasmit Das
"""
import cv2
import numpy as np

###########################################
# Function to detect and compute keypoints using SIFT is implemented here
def SIFT_detector(img,f_count,sig,layers):
	
	img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# creating a sift detector
	SIFT = cv2.xfeatures2d.SIFT_create(nfeatures=f_count,nOctaveLayers=layers,contrastThreshold=0.03,edgeThreshold=10,sigma=sig)
	# Applying Sift detector to the image
	keypoints, descriptor = SIFT.detectAndCompute(img_gray,None)
	#outImg=0
	# Plotting the keypoints along with their scale not required for this case
	#img = cv2.drawKeypoints(img, keypoints,outImg,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	pts=[]
	# Obtaining keypoints from the object keypoint
	for keypoint in keypoints:
		pts.append([np.round(keypoint.pt[0],0),np.round(keypoint.pt[1],0)])	
	# Returning the plotted image, keypoints and the descriptors
	return img,np.asarray(pts,dtype='int'),np.asarray(descriptor)

###########################################

###########################################
# Function to find the Normalised Cross Corelation for each level starts here
def NCC(kp1,des1,kp2,des2,T_ncc,T_max_global):
        """ Code for NCC to correlate between two views of the same image
        """ 
	src_mat=np.zeros((len(kp1),len(kp2)),dtype='float')
	for loop1 in range(0,len(kp1)):
		for loop2 in range(0,len(kp2)):
			src_mean=np.mean(des1[loop1,:])
			dest_mean=np.mean(des2[loop2,:])
			src_norm=np.subtract(des1[loop1,:],src_mean)
			dest_norm=np.subtract(des2[loop2,:],dest_mean)
			num=np.sum(np.multiply(src_norm,dest_norm))
			src_sq=np.sum(np.square(src_norm))
			dest_sq=np.sum(np.square(dest_norm))
			den=np.sqrt(src_sq*dest_sq)
			src_mat[loop1,loop2]=num/den
	pts=[]
	
	# Eliminate weak correspondences
	for loop1 in range(0,len(kp1)):
		for loop2 in range(0,len(kp2)):
			if src_mat[loop1,loop2]==np.max(src_mat[loop1,:]) and src_mat[loop1,loop2]>T_max_global*np.mean(src_mat):
				loc_max=src_mat[loop1,loop2]
				src_mat[loop1,loop2]=np.min(src_mat[loop1,:])
				# Eliminate false positive correspondences by thresholding 
				if np.max(src_mat[loop1,:])/loc_max < T_ncc:
				#if loc_max/np.max(src_mat[loop1,:]) > T_ncc:
					#Removing Many to one Correspondence
					src_mat[:,loop2]=0
					src_mat[loop1,loop2]=loc_max; print src_mat[loop1,loop2]
					pts.append([kp1[loop1,0],kp1[loop1,1],kp2[loop2,0],kp2[loop2,1]])
					
        return np.asarray(pts)
# Function to find the Normalised Cross Corelation for each level ends here
###########################################
