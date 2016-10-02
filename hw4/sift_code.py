# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 20:13:30 2016

@author: debasmit
"""
import cv2
import numpy as np


def imgSIFT(path):
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #PeakThreshold=3
    SIFT = cv2.xfeatures2d.SIFT_create(nfeatures=1000,nOctaveLayers=4,contrastThreshold=0.12,edgeThreshold=10,sigma=1.6)
    dummy = np.zeros((1,1))
    kp, des = SIFT.detectAndCompute(gray,None) #This returns the keypoint pixels and their descriptors
    img = cv2.drawKeypoints(img, kp,dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpix =[]
    i=0
    for k in kp:
        kpix.append([k.pt[0],k.pt[1]])
        i=i+1   
    
    kpix=np.array(kpix)
    kpix=np.round(kpix)
    return img,kpix,des
    #The image is returned with the image containing the sift features , the pixel locations and the 128-point descriptor

def SSDimg(l,ssd,tssd):
     #This is used for plotting the concatenated images
    h1, w1 = l[0][0].shape[:2]
    h2, w2 = l[1][0].shape[:2]
    img=np.zeros((max(h1,h2),w1+w2,3))
    img[:h1, :w1]=l[0][0]
    img[:h2, w1:w1+w2]=l[1][0]    
    
    if l[0][1].shape[0]>=l[1][1].shape[0]:
        maxi=0;
        mini=1;
    else:
        maxi=1;
        mini=0;
        
    nmin=l[mini][1].shape[0]
    nmax=l[maxi][1].shape[0]    
    # This is used to create SSD Matrix for feature points between 2 images
    sift_mat=np.zeros((nmin,nmax))
    for i in range(0,nmin): 
        for j in range(0,nmax):
            sift_mat[i,j]=np.linalg.norm((l[mini][2][i,:]-l[maxi][2][j,:]))
                
    SSD=np.square(sift_mat)
    # This is done to choose significant feature points and to remove many-to-one mappings
    for i in range(0,nmin): 
       for j in range(0,nmax):
            if SSD[i,j]==np.amin(SSD[i,:]) and SSD[i,j]<tssd*np.min(SSD):
                loc_min=SSD[i,j]
                SSD[i,j]=np.amax(SSD[i,:])
                if loc_min/np.amin(SSD[i,:])<ssd:
                    if mini==0:
                        pt1=l[0][1][i]
                        pt2=l[1][1][j]+np.array([l[0][0].shape[1],0])
                        cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (0,255,0))
                    else:
                        pt1=l[0][1][j]
                        pt2=l[1][1][i]+np.array([l[0][0].shape[1],0]) 
                        cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (0,255,0))                      
                              
    return img
    #The img showing the concatenated images and SSD matched images is output        

def NCCimg(l,ncc,tncc):
     #This is used for plotting the concatenated images
    h1, w1 = l[0][0].shape[:2]
    h2, w2 = l[1][0].shape[:2]
    img=np.zeros((max(h1,h2),w1+w2,3))
    img[:h1, :w1]=l[0][0]
    img[:h2, w1:w1+w2]=l[1][0]    
#    
    if l[0][1].shape[0]>=l[1][1].shape[0]:
        maxi=0;
        mini=1;
    else:
        maxi=1;
        mini=0;
   
       
    nmin=l[mini][1].shape[0]
    nmax=l[maxi][1].shape[0]    
    # This is used to create NCC Matrix for feature points between 2 images
    NCC=np.zeros((nmin,nmax))
    for i in range(0,nmin): 
        for j in range(0,nmax):
            f1=l[mini][2][i,:]-np.mean(l[mini][2][i,:])
            f2=l[maxi][2][j,:]-np.mean(l[maxi][2][j,:])
            NCC[i,j]=np.sum(np.multiply(f1,f2))/(((np.linalg.norm(f1)*np.linalg.norm(f2)))**0.5)
                
   
    # This is done to choose significant feature points and to remove many-to-one mappings
    for i in range(0,nmin): 
       for j in range(0,nmax):
            if NCC[i,j]==np.amax(NCC[i,:]) and NCC[i,j]>tncc*np.max(NCC):
                loc_max=NCC[i,j]
                NCC[i,j]=np.amin(NCC[i,:])
                if loc_max/np.max(NCC[i,:])>ncc:
                    if mini==0:
                        pt1=l[0][1][i]
                        pt2=l[1][1][j]+np.array([l[0][0].shape[1],0])
                        cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (0,255,0))
                    else:
                        pt1=l[0][1][j]
                        pt2=l[1][1][i]+np.array([l[0][0].shape[1],0]) 
                        cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (0,255,0))                      
                              
    return img                     
    #The img showing the concatenated images and NCC matched images is output 


if __name__ == "__main__":
    l1=imgSIFT('pair2/1.jpg')
    l2=imgSIFT('pair2/2.jpg')
    l=[l1,l2]
    img1=NCCimg(l,0.7,0.9)
    img2=SSDimg(l,0.7,5)
    cv2.imwrite('sift_results/SIFTResultNCC.jpg',img1)
    cv2.imwrite('sift_results/SIFTResultSSD.jpg',img2)    