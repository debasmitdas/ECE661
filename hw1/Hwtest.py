# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:54:58 2016

@author: vishwa
"""

import cv2
import numpy as np
path = "/home/vishwa/661/PicsHw2"

image_1 = cv2.imread(path+"/1.jpg")
image_2 = cv2.imread(path+"/2.jpg")
image_3 = cv2.imread(path+"/3.jpg")
image_4 = cv2.imread(path+"/Seinfeld.jpg")
#cv2.waitKey(550)
# points obtained from gimp
Points_1a1=array([[421,2108,1],[531,3303,1],[1495,2148,1],[1357,3312,1]])
Points_1b=array([[795,1590,1],[768,2983,1],[1597,1616,1],[1518,2987,1]])
Points_1c=array([[562,999,1],[421,2425,1],[1412,1026,1],[1478,2406,1]])
Points_1d=array([[0,0,1],[0,2560,1],[1536,0,1],[1536,2560,1]])

Points_1a=

#H=array([[a,b,c],[d,e,f],[g,h,i]])
# Find the Homography between Seinfeld image and 1b
homo_trans_1db=np.dot(np.linalg.pinv(Points_1d),Points_1b)
homography_1db=homo_trans_1db.transpose()

# Find the Homography between 1b and 1a
homo_trans_1ba=np.dot(np.linalg.pinv(Points_1b),Points_1a)
homography_1ba=homo_trans_1ba.transpose()

# Find the Homography between 1b and 1c
homo_trans_1bc=np.dot(np.linalg.pinv(Points_1b),Points_1c)
homography_1bc=homo_trans_1bc.transpose()

tmp_image=image_2
for i in range(0,image_4.shape[0]-1):
    for j in range (0,image_4.shape[1]-1):
        tmp_xy=[i,j,1]
        trans_val=np.dot(homography_1db,tmp_xy)
        tmp_image[trans_val[0]][trans_val[1]]=image_4[i][j]
        
# Write the output in a new file
#cv2.namedWindow('test',cv2.WINDOW_NORMAL)
#cv2.imshow('test',tmp_image)  

# Write the output in a new file
cv2.imwrite('test_1b.jpg',tmp_image)
new_image2=tmp_image

# Transform new 1b to 1a
tmp_image=image_1
for i in range(795,1596):
    for j in range (1616,2982):
        tmp_xy=[i,j,1]
        trans_val=np.dot(homography_1ba,tmp_xy)
        tmp_image[trans_val[0]][trans_val[1]]=new_image2[i][j]
        
# Write the output in a new file
#cv2.namedWindow('test',cv2.WINDOW_NORMAL)
#cv2.imshow('test',tmp_image)  

# Write the output in a new file
cv2.imwrite('test_1a.jpg',tmp_image)

tmp_image=image_3
for i in range(795,1596):
    for j in range (1616,2982):
        tmp_xy=[i,j,1]
        trans_val=np.dot(homography_1bc,tmp_xy)
        tmp_image[trans_val[0]][trans_val[1]]=new_image2[i][j]
        
# Write the output in a new file
#cv2.namedWindow('test',cv2.WINDOW_NORMAL)
#cv2.imshow('test',tmp_image)  

# Write the output in a new file
cv2.imwrite('test_1c.jpg',tmp_image)

# Find the Homography between Seinfeld image and 1a
homo_trans_1da=np.dot(np.linalg.pinv(Points_1d),Points_1a)
homography_1da=homo_trans_1da.transpose()

tmp_image=image_1
for i in range(0,image_4.shape[0]-1):
    for j in range (0,image_4.shape[1]-1):
        tmp_xy=[i,j,1]
        trans_val=np.dot(homography_1da,tmp_xy)
        tmp_image[trans_val[0]][trans_val[1]]=image_4[i][j]
# Write the output in a new file
cv2.imwrite('test_1da.jpg',tmp_image)

# Find the Homography between Seinfeld image and 1b
homo_trans_1da=np.dot(np.linalg.pinv(Points_1d),Points_1c)
homography_1da=homo_trans_1da.transpose()

tmp_image=image_3
for i in range(0,image_4.shape[0]-1):
    for j in range (0,image_4.shape[1]-1):
        tmp_xy=[i,j,1]
        trans_val=np.dot(homography_1da,tmp_xy)
        tmp_image[trans_val[0]][trans_val[1]]=image_4[i][j]
# Write the output in a new file
cv2.imwrite('test_1dc.jpg',tmp_image)

