"""
Author : Debasmit Das
"""

import cv2
import SIFT as S
import RANSAC as R
import Image_Mapping as IM
import numpy as np
import Dogleg as Dg
#########################################
# Obtaining input from the user
#########################################
"""
path = "Enter the full path for the images:"
img_name = "Enter the image name with file type/ file extension:"
sigma_prompt = "Enter the sigma (Kernel size):"
F_prompt = "Enter the no. of features in which SIFT needs to be detected:"
Layers_prompt = "Enter the no. of octave layers for which keypoints needs to be computed:"
T_ncc_prompt="Enter the value of threshold for NCC metrics (0.6~0.9)"
T_max_global_prompt="Enter the value of threshold for finding keypoint among global maximums \n Lower the value to get more keypoints... (0.2~0.8)  "
error_prompt="Enter the percentage error in the correspondence"
pts_per_trial_prompt="Enter the samples to be considered in each trial( Value between 5~10 ):"
percent_prob_prompt="Enter the probability that atleast one trial is outlier free( Value between 99~100 ):"
G_Noise_prompt="Enter the gaussian noise in the images(Value between 0.5 ~ 2):"
file_path=raw_input(">"+path)
img_name1=raw_input(">"+img_name)
img_name2=raw_input(">"+img_name)
sigma=float(raw_input(">"+sigma_prompt))
f_count=int(raw_input(">"+F_prompt))
layers=int(raw_input(">"+Layers_prompt))
T_ncc=int(raw_input(">"+T_ncc_prompt))
T_max_global=int(raw_input(">"+T_max_global_prompt))
error=float(raw_input(">"+error_prompt))/100
pts_per_trial=int(raw_input(">"+pts_per_trial_prompt))
percent_prob=float(raw_input(">"+percent_prob_prompt))/100
G_Noise=float(raw_input(">"+G_Noise_prompt))
#########################################
# Obtaining input from the user ends here
#########################################
"""
###########################################
# Sample input code for referrence
###########################################
img_1=cv2.imread('1.jpg')
img_2=cv2.imread('2.jpg')
img_3=cv2.imread('3.jpg')
img_4=cv2.imread('4.jpg')
img_5=cv2.imread('5.jpg')

layers=4
sigma=4
T_ncc=0.66
T_max_global=0.96
f_count=400
error=0.4
pts_per_trial=6
percent_prob=0.999
G_Noise=1
"""
###########################################

###########################################
# Computing SIFT keypoints and Descriptor
###########################################
"""

img1, kp1, desp1 = S.SIFT_detector(img_1,f_count,sigma,layers)
img2, kp2, desp2 = S.SIFT_detector(img_2,f_count,sigma,layers)
img3, kp3, desp3 = S.SIFT_detector(img_3,f_count,sigma,layers)
img4, kp4, desp4 = S.SIFT_detector(img_4,f_count,sigma,layers)
img5, kp5, desp5 = S.SIFT_detector(img_5,f_count,sigma,layers)
###########################################
# Computing correspondence using NCC
###########################################
Coord_NCC12=S.NCC(kp1,desp1,kp2,desp2,T_ncc,T_max_global)
img=IM.plotting(Coord_NCC12,img1,img2)
cv2.imwrite("Pair1.jpg",img)
Coord_NCC23=S.NCC(kp2,desp2,kp3,desp3,T_ncc,T_max_global)
img=IM.plotting(Coord_NCC23,img2,img3)
cv2.imwrite("Pair2.jpg",img)
Coord_NCC34=S.NCC(kp3,desp3,kp4,desp4,T_ncc,T_max_global)
img=IM.plotting(Coord_NCC34,img3,img4)
cv2.imwrite("Pair3.jpg",img)
Coord_NCC45=S.NCC(kp4,desp4,kp5,desp5,T_ncc,T_max_global)
img=IM.plotting(Coord_NCC45,img4,img5)
cv2.imwrite("Pair4.jpg",img)
###########################################

###########################################
# Computing Homography using RANSAC
###########################################
pts_src12=Coord_NCC12[:,0:2]
pts_dest12=Coord_NCC12[:,2:4]
H_coarse12=R.Ransac(pts_src12,pts_dest12,G_Noise,error,pts_per_trial,percent_prob)
img=IM.outlier_plotting(Coord_NCC12,img1,img2,H_coarse12,G_Noise)
cv2.imwrite("Pair1outliers.jpg",img)

pts_src23=Coord_NCC23[:,0:2]
pts_dest23=Coord_NCC23[:,2:4]
H_coarse23=R.Ransac(pts_src23,pts_dest23,G_Noise,error,pts_per_trial,percent_prob)
img=IM.outlier_plotting(Coord_NCC23,img2,img3,H_coarse23,G_Noise)
cv2.imwrite("Pair2outliers.jpg",img)

pts_src34=Coord_NCC34[:,0:2]
pts_dest34=Coord_NCC34[:,2:4]
H_coarse34=R.Ransac(pts_src34,pts_dest34,G_Noise,error,pts_per_trial,percent_prob)
img=IM.outlier_plotting(Coord_NCC34,img3,img4,H_coarse34,G_Noise)
cv2.imwrite("Pair3outliers.jpg",img)

pts_src45=Coord_NCC45[:,0:2]
pts_dest45=Coord_NCC45[:,2:4]
H_coarse45=R.Ransac(pts_src45,pts_dest45,G_Noise,error,pts_per_trial,percent_prob)
img=IM.outlier_plotting(Coord_NCC45,img4,img5,H_coarse45,G_Noise)
cv2.imwrite("Pair4outliers.jpg",img)

###########################################



###########################################
# Image Mosaicing
###########################################
# considering the image 3 to be the center image

H_coarse13=H_coarse12.dot(H_coarse23)

H_coarse35=H_coarse34.dot(H_coarse45)
H_coarse53=np.linalg.inv(H_coarse35)
H_coarse43=np.linalg.inv(H_coarse34)
H_33=np.array([[1,0,0],[0,1,0],[0,0,1]])
# boundaries of each image in plane of image 3
B1=IM.Boundary(H_coarse13/H_coarse13[2,2],img_1)
B2=IM.Boundary(H_coarse23/H_coarse23[2,2],img_2)
B3=np.array([[0,0],[0,img_3.shape[1]],[img_3.shape[0],0],[img_3.shape[0],img_3.shape[1]]])
B3=B3.T
B4=IM.Boundary(H_coarse43/H_coarse43[2,2],img_4)
B5=IM.Boundary(H_coarse53/H_coarse53[2,2],img_5)
min_xy=np.amin(np.amin([B1,B2,B3,B4,B5],2),0)
max_xy=np.amax(np.amax([B1,B2,B3,B4,B5],2),0)


offset_dim=max_xy-min_xy
Base_image=np.zeros((offset_dim[1],offset_dim[0],3),dtype='uint')
# Mapping the Homographies into the image on the required plane
output=IM.image_mapping(Base_image,img_1,np.linalg.inv(H_coarse13/H_coarse13[2,2]),min_xy)
cv2.imwrite("FinalImage_WithoutDogleg_WOT_Blending1.jpg",output)
output=IM.image_mapping(output,img_2,np.linalg.inv(H_coarse23/H_coarse23[2,2]),min_xy)
cv2.imwrite("FinalImage_WithoutDogleg_WOT_Blending2.jpg",output)
output=IM.image_mapping(output,img_3,np.linalg.inv(H_33),min_xy)
cv2.imwrite("FinalImage_WithoutDogleg_WOT_Blending3.jpg",output)
output=IM.image_mapping(output,img_4,H_coarse34/H_coarse34[2,2],min_xy)
cv2.imwrite("FinalImage_WithoutDogleg_WOT_Blending4.jpg",output)
output=IM.image_mapping(output,img_5,H_coarse35/H_coarse35[2,2],min_xy)
cv2.imwrite("FinalImage_WithoutDogleg_WOT_Blending5.jpg",output)
###########################################

Inliersset1=R.Inliers(pts_src12,pts_dest12,H_coarse12,3*G_Noise)
Inliersset2=R.Inliers(pts_src23,pts_dest23,H_coarse23,3*G_Noise)
Inliersset3=R.Inliers(pts_src34,pts_dest34,H_coarse34,3*G_Noise)
Inliersset4=R.Inliers(pts_src45,pts_dest45,H_coarse45,3*G_Noise)
###########################################
# Fine Tuning Homography using Dogleg
###########################################
H_finetune12=Dg.Dogleg(Inliersset1[:,1],Inliersset1[:,0],H_coarse12)
H_finetune23=Dg.Dogleg(Inliersset2[:,1],Inliersset2[:,0],H_coarse23)
H_finetune34=Dg.Dogleg(Inliersset3[:,1],Inliersset3[:,0],H_coarse34)
H_finetune45=Dg.Dogleg(Inliersset4[:,1],Inliersset4[:,0],H_coarse45)

H_finetune13=H_finetune12.dot(H_finetune23)
H_finetune13=H_finetune13/H_finetune13[2,2]
H_finetune35=H_finetune34.dot(H_finetune45)
H_finetune35=H_finetune35/H_finetune35[2,2]
H_finetune43=np.linalg.inv(H_finetune34)
H_finetune53=np.linalg.inv(H_finetune35)
##########################################

# boundaries of each image in plane of image 3 after Dogleg
B1_dg=IM.Boundary(H_finetune13,img_1)
B2_dg=IM.Boundary(H_finetune23,img_2)
B3_dg=np.array([[0,0],[0,img_3.shape[1]],[img_3.shape[0],0],[img_3.shape[0],img_3.shape[1]]])
B3_dg=B3_dg.T
B4_dg=IM.Boundary(H_finetune43,img_4)
B5_dg=IM.Boundary(H_finetune53,img_5)
min_xy_dg=np.amin(np.amin([B1_dg,B2_dg,B3_dg,B4_dg,B5_dg],2),0)
max_xy_dg=np.amax(np.amax([B1_dg,B2_dg,B3_dg,B4_dg,B5_dg],2),0)

offset_dim_dg=max_xy_dg-min_xy_dg
Base_image_dg=np.zeros((offset_dim_dg[1],offset_dim_dg[0],3),dtype='uint')
# Mapping the Homographies and all images into the image on the required plane with dogleg optimisation
output1=IM.image_mapping(Base_image_dg,img_1,np.linalg.inv(H_finetune13),min_xy_dg)
cv2.imwrite('FinalImage_WithDogleg1.jpg',output1)
output1=IM.image_mapping(output1,img_2,np.linalg.inv(H_finetune23),min_xy_dg)
cv2.imwrite('FinalImage_WithDogleg2.jpg',output1)
output1=IM.image_mapping(output1,img_3,np.linalg.inv(H_33),min_xy_dg)
cv2.imwrite('FinalImage_WithDogleg3.jpg',output1)
output1=IM.image_mapping(output1,img_4,H_finetune34,min_xy_dg)
cv2.imwrite('FinalImage_WithDogleg4.jpg',output1)
output1=IM.image_mapping(output1,img_5,H_finetune35,min_xy_dg)
cv2.imwrite('FinalImage_WithDogleg5.jpg',output1)

###########################################################
# Parameter for Image - Baxter

layers=4
sigma=4
T_ncc=0.65
T_max_global=0.9
f_count=400
error=0.3
pts_per_trial=6
percent_prob=0.999
G_Noise=1
"""
#########################################################


###########################################################
# Parameter for Image - ECE
"""
#layers=4
#sigma=4
#T_ncc=0.65
#T_max_global=0.9
#f_count=400
#error=0.3
#pts_per_trial=6
#percent_prob=0.999
#G_Noise=1
"""

###########################################################
# Parameter for Image - Imageset3
"""
#layers=4
#sigma=4
#T_ncc=0.65
#T_max_global=0.95
#f_count=400
#error=0.4
#pts_per_trial=7
#percent_prob=0.999
#G_Noise=1
