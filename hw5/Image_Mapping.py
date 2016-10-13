
"""
Author: Debasmit Das
"""
import math
import numpy as np
import cv2
####################################
# Getting RGB data function using weighted average starts here
####################################
def getdata(point, img):
    tp_left =img[(math.floor(point[1])),(math.floor(point[0]))]
    tp_right =img[math.floor(point[1]),math.floor(point[0]+1)]
    bt_left =img[math.floor(point[1]+1),math.floor(point[0])]
    bt_right =img[math.floor(point[1]+1),math.floor(point[0]+1)]
    diff_x = point[1] - math.floor(point[1])
    diff_y = point[0] - math.floor(point[0])
    tp_left_weight= pow(pow(diff_x,2)+pow(diff_y,2),-0.5)
    tp_right_weight = pow(pow(diff_x,2)+pow(1-diff_y,2),-0.5)
    bt_left_weight = pow(pow(1-diff_x,2)+pow(diff_y,2),-0.5)
    bt_right_weight = pow(pow(1-diff_x,2)+pow(1-diff_y,2),-0.5)
    resultant_pt = (tp_left*tp_left_weight+tp_right*tp_right_weight+bt_left*bt_left_weight+bt_right*bt_right_weight)/(tp_left_weight+tp_right_weight+bt_left_weight+bt_right_weight)
    return resultant_pt
####################################
# getting RGB data function using weighted average ends here
####################################
    
####################################    
# Image mapping code starts here
####################################
def image_mapping(src_image,dest_image,Homography,offset_xy):
    
    for i in range(0,src_image.shape[0]):
        for j in range(0,src_image.shape[1]):
                point_tmp = np.array([j+offset_xy[0],i+offset_xy[1], 1])
                trans_coord = np.array(np.dot(Homography,point_tmp))
                trans_coord = trans_coord/trans_coord[2]
                
                if (trans_coord[1]>0) and (trans_coord[1]<dest_image.shape[0]-1) and (trans_coord[0]>0) and (trans_coord[0]<dest_image.shape[1]-1):
                    src_image[i][j]=getdata(trans_coord,dest_image)
    return src_image
####################################
# Image mapping code ends here.
####################################

####################################    
# Image boundary code starts here
####################################
def Boundary(H,src_image):
    # Boundary points of the given image is extracted here
    Points=np.array([[0,0,1],[0,src_image.shape[1],1],[src_image.shape[0],0,1],[src_image.shape[0],src_image.shape[1],1]])
    tmp=np.zeros((Points.shape[1],Points.shape[0]))
    # Boundary points of the given image on the new plane is computed here
    tmp=np.array((np.dot(H,Points.T)).T)
    for i in range(0,Points.shape[0]):
        tmp[i]=tmp[i]/tmp[i,2]
    tmp=tmp.T
    return tmp[0:2,:]
####################################    
# Image boundary code ends here
####################################
    
####################################    
# Image corresponding plotting code starts here
####################################   
def plotting(Corner_Coord,img1,img2):
	# creating a base image which has image 1 and image 2
    	img=np.zeros((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3))
    	img[:img1.shape[0], :img1.shape[1]]=img1
    	img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]]=img2
	
	# plotting the correspondence using lines
	for coord in Corner_Coord:
		img=cv2.line(img,(coord[0],coord[1]),(img1.shape[1]+coord[2],coord[3]), (0,255,0))
	return img
####################################    
# Image corresponding plotting code ends here
####################################   
 
 
####################################    
# Image outlier plotting code starts here
####################################   
def outlier_plotting(Corner_Coord,img1,img2,H,sigma):
	# creating a base image which has image 1 and image 2
	img=np.zeros((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3))
	img[:img1.shape[0], :img1.shape[1]]=img1
	img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]]=img2
	pts_src=Corner_Coord[:,0:2]
	pts_dest=Corner_Coord[:,2:4]
	
	for loop in range(len(pts_src)):
		pts_dest_calc=np.dot(H,[pts_src[loop,0],pts_src[loop,1],1])
		pts_dest_calc=pts_dest_calc/pts_dest_calc[2]
		pts_diff=np.sqrt(np.sum((pts_dest_calc[0:2]-pts_dest[loop,:])**2))
		if  pts_diff < 3*sigma: 
			# plotting the correspondence using lines
			cv2.circle(img,(pts_src[loop,0],pts_src[loop,1]),2,(255,0,0),2)
			cv2.circle(img,(img1.shape[1]+int(pts_dest_calc[0]),int(pts_dest_calc[1])),2,(255,0,0),2)
			cv2.line(img,(pts_src[loop,0],pts_src[loop,1]),(img1.shape[1]+int(pts_dest_calc[0]),int(pts_dest_calc[1])), (0,255,0))
		else :
			cv2.circle(img,(pts_src[loop,0],pts_src[loop,1]),2,(0,0,255),2)
			cv2.circle(img,(img1.shape[1]+int(pts_dest_calc[0]),int(pts_dest_calc[1])),2,(0,0,255),2)
			cv2.line(img,(pts_src[loop,0],pts_src[loop,1]),(img1.shape[1]+int(pts_dest_calc[0]),int(pts_dest_calc[1])), (0,0,255))
	return img
####################################    
# Image outlier plotting code ends here
####################################   
