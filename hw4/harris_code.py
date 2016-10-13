# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:16:21 2016

@author: debasmit
"""

#Code for harris detector


import cv2
import numpy as np
import math

def harris(img,sigma,level,k,th=0.3):
	""" function for harris corner detection 
	"""
 
	# We find the size of the Haar Filter 
        Wsize=int(math.ceil(math.ceil(4*level*sigma)/2)*2)
        HaarX=np.concatenate((-np.ones((Wsize,Wsize/2)), np.ones((Wsize,Wsize/2))),axis=1)
        HaarY=np.concatenate((np.ones((Wsize/2,Wsize)), -np.ones((Wsize/2,Wsize))),axis=0)
        
        Ix=cv2.filter2D(img,-1,HaarX)
        Iy=cv2.filter2D(img,-1,HaarY)
        ht=img.shape[0]
        wd=img.shape[1]
        newW= int(math.ceil(math.ceil(5*level*sigma)/2)*2)+1
        newhalfW= int(math.ceil(newW/2))
        Cresp=np.zeros((img.shape),dtype='float')
	for i in range(newhalfW,ht-newhalfW+1):
	  for j in range(newhalfW,wd-newhalfW+1):
             Ix_ij=Ix[i-newhalfW:i+int(math.floor(newW/2)),j-newhalfW:j+int(math.floor(newW/2))]
             Iy_ij=Iy[i-newhalfW:i+int(math.floor(newW/2)),j-newhalfW:j+int(math.floor(newW/2))]
             Mat=np.zeros((2,2),dtype='float')
             Mat[0,0]=np.sum(np.square(Ix_ij))
             Mat[1,1]=np.sum(np.square(Iy_ij))
             Mat[0,1]=np.sum(np.multiply(Ix_ij,Iy_ij)); Det=np.linalg.det(Mat);tr=np.trace(Mat);Cresp[i,j]=Det-k*(tr**2)
	         
    # Till here, the corner response of the images are found         
	         
	
	#Non-maximal supression is carried out to remove multiple corners
     	
	Wsup=35
	corners=[]
	
     	for i in range(Wsup/2,ht-Wsup/2+1):
         for j in range(Wsup/2,wd-Wsup/2+1):
		Crespsub=Cresp[i-Wsup/2:i+Wsup/2+1,j-Wsup/2:j+Wsup/2+1]
		if Cresp[i,j]==np.max(Crespsub) and Cresp[i,j]> th*np.max(np.abs(Cresp)):
			corners.append([i,j])
	return np.asarray(corners)


def plotimg(corners,img1,img2):
	    #This function is used for plotting the concatenated images and the matched points
    	img=np.zeros((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3))
    	img[:img1.shape[0], :img1.shape[1]]=img1
    	img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]]=img2
	out=img

	for coord in corners:
		cv2.circle(out,(coord[1],coord[0]),5,(255,0,0),2)
		cv2.circle(out,(img1.shape[1]+coord[3],coord[2]),5,(255,0,0),2)
		cv2.line(out,(coord[1],coord[0]),(img1.shape[1]+coord[3],coord[2]), (0,255,0))
	return out


def SSD(C1,C2,img1,img2,W,ssd,tssd):
        """ Function for SSD to match points between feature points of 2 images and return the corresponding points
        """
	W_half=int(W/2);
	if img1.shape[0]>img2.shape[0] or img1.shape[1]>img2.shape[1]:
		
		src=C2
		dest=C1
		src_img=img2
		dest_img=img1
		fl=1
	else:
		
		src=C1
		dest=C2
		src_img=img1
		dest_img=img2
		fl=0
  #SSD Matrix for different feature points of 2 images
	ssdmat=np.zeros((len(src),len(dest)),dtype='float')
	for i in range(0,len(src)):
		for j in range(0,len(dest)):
			src_local=src_img[src[i,0]-W_half:src[i,0]+W_half+1,src[i,1]-W_half:src[i,1]+W_half+1]
			dest_local=dest_img[dest[j,0]-W_half:dest[j,0]+W_half+1,dest[j,1]-W_half:dest[j,1]+W_half+1]
			ssdmat[i,j]=np.sum(np.square(np.subtract(src_local,dest_local)))
	pts=[]
	
	# This is done to choose significant feature points and to remove many-to-one mappings
	for i in range(0,len(src)):
		for j in range(0,len(dest)):
			if ssdmat[i,j]==np.min(ssdmat[i,:]) and ssdmat[i,j]<tssd*np.mean(ssdmat[:,:]):
				loc_min=ssdmat[i,j]
				ssdmat[i,j]=np.max(ssdmat[i,:])
				if loc_min/np.min(ssdmat[i,:])<ssd:
					ssdmat[:,j]=np.max(ssdmat)
					ssdmat[i,j]=loc_min					
					if fl==0:
						pts.append([src[i,0],src[i,1],dest[j,0],dest[j,1]])
					elif fl == 1:
						pts.append([dest[j,0],dest[j,1],src[i,0],src[i,1]])       
        
        return np.asarray(pts)

def NCC(C1,C2,img1,img2,W,ncc,tncc):
        """ Function for NCC to match points between feature points of 2 images and return the corresponding points
        """
	W_half=int(W/2);
	if img1.shape[0]>img2.shape[0] or img1.shape[1]>img2.shape[1]:
		
		src=C2
		dest=C1
		src_img=img2
		dest_img=img1
		fl=1
	else:
		
		src=C1
		dest=C2
		src_img=img1
		dest_img=img2
		fl=0
   #NCC Matrix for different feature points of 2 images
	nccmat=np.zeros((len(src),len(dest)),dtype='float')
	for i in range(0,len(src)):
		for j in range(0,len(dest)):
			src_local=src_img[src[i,0]-W_half:src[i,0]+W_half+1,src[i,1]-W_half:src[i,1]+W_half+1]
			dest_local=dest_img[dest[j,0]-W_half:dest[j,0]+W_half+1,dest[j,1]-W_half:dest[j,1]+W_half+1]
			src_mean=np.mean(src_local)
			dest_mean=np.mean(dest_local)
			src_norm=np.subtract(src_local,src_mean)
			dest_norm=np.subtract(dest_local,dest_mean)
			num=np.sum(np.multiply(src_norm,dest_norm))
			src_sq=np.sum(np.square(src_norm))
			dest_sq=np.sum(np.square(dest_norm))
			den=np.sqrt(src_sq*dest_sq)
			nccmat[i,j]=num/den
	pts=[]
	
	# This is done to choose significant feature points and to remove many-to-one mappings
	for i in range(0,len(src)):
		for j in range(0,len(dest)):
			if nccmat[i,j]==np.max(nccmat[i,:]) and nccmat[i,j]>tncc*np.max(nccmat[:,:]):
				loc_max=nccmat[i,j]
				nccmat[i,j]=np.min(nccmat[i,:])
				if loc_max/np.max(nccmat[i,:])>ncc:
					nccmat[:,j]=np.min(nccmat)
					nccmat[i,j]=loc_max					
					if fl==0:
						pts.append([src[i,0],src[i,1],dest[j,0],dest[j,1]])
					elif fl == 1:
						pts.append([dest[j,0],dest[j,1],src[i,0],src[i,1]])
					
        return np.asarray(pts)


if __name__ == "__main__":

    
    img1=cv2.imread("pair1/1.jpg")
    img2=cv2.imread("pair1/2.jpg")
    level=1
    sigma=1.4
    W=24
    k=0.06
    ncc=0.9
    tncc=0.7
    ssd=0.8
    tssd=5

    img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    C1 = harris(img1gray,sigma,level,k,0.3)
    C2 = harris(img2gray,sigma,level,k,0.3)
    
    # Obtaining matches using NCC metrics
    Cord = NCC(C1,C2,img1gray,img2gray,W,ncc,tncc)
    cv2.imwrite('harris_results/HarrisNCCl1.jpg',plotimg(Cord,img1,img2))
    
    
    # Obtaining matches using SSD metrics
    Cord = SSD(C1,C2,img1gray,img2gray,W,ssd,tssd)
    cv2.imwrite('harris_results/HarrisSSDl1.jpg',plotimg(Cord,img1,img2))
    
