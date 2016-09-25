# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 20:13:30 2016

@author: debasmit
"""
import cv2
import numpy as np

#cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → None¶ # Command for drawing lines in images
def imgSIFT(path):
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #default
    #SIFT = cv2.xfeatures2d.SIFT_create(nfeatures=100,nOctaveLayers=3,contrastThreshold=0.04,edgeThreshold=10,sigma=1.6)
    #PeakThreshold=3
    SIFT = cv2.xfeatures2d.SIFT_create(nfeatures=500,nOctaveLayers=1,contrastThreshold=0.12,edgeThreshold=10,sigma=1.6)
    dummy = np.zeros((1,1))
    kp, des = SIFT.detectAndCompute(gray,None)
    img = cv2.drawKeypoints(img, kp,dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpix =[]
    i=0
    for k in kp:
        kpix.append([k.pt[0],k.pt[1]])
        i=i+1   
    #print i
    kpix=np.array(kpix)
    kpix=np.round(kpix)
    return img,kpix,des

#def SSD(img1,kpix1,img2,kpix2)    
#def Euclidean(img1,kpix1, des1, img2, kpix2, des2):
def Euclidean(l,euc):    
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
    
    sift_mat=np.zeros((nmin,nmax))
    for i in range(0,nmin): 
        for j in range(0,nmax):
            sift_mat[i,j]=np.linalg.norm((l[mini][2][i,:]-l[maxi][2][j,:]))
    
    
    
    for i in range(0,nmin): 
        min_j=0;
        mindist=np.inf       
        for j in range(0,nmax):
            if sift_mat[i,j]==np.amin(sift_mat[i,:]) and sift_mat[i,j]<5*np.amin(np.amin(sift_mat)):
                loc_min=sift_mat[i,j]
                sift_mat[i,j]=np.amax(sift_mat[i,:])
                if loc_min/np.amin(sift_mat[i,:])<euc:
                    if mini==0:
                        pt1=l[0][1][i]
                        pt2=l[1][1][j]+np.array([l[0][0].shape[1],0])
                        cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (0,255,0))
                    else:
                        pt1=l[0][1][j]
                        pt2=l[1][1][i]+np.array([l[0][0].shape[1],0]) 
                        cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (0,255,0))                      
                              
    return img                         
    
if __name__ == "__main__":
    l1=imgSIFT('mypair/1.jpg')
    l2=imgSIFT('mypair/2.jpg')
    l=[l1,l2]
    img=Euclidean(l,0.7)
    cv2.imwrite('sift_results/SIFTResult.jpg',img)