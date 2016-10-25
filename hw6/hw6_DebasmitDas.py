# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:46:23 2016

@author: debasmit
"""

import numpy as np
import cv2

    


#Applying OTSU's algorithm for RGB images
def otsuRGB(img, maskinvert, iterations):
    #Mask invert is just for inverting and non-inverting channels 
    output=np.zeros((img.shape[0],img.shape[1]),np.uint8)
    output.fill(255)
    
    #Iterating over all the three channels
    for c in range(0,3):
        
        #The mask of channel c
        mask=None        
        
        #Applying Otsu's algorithm to each channel for a number of iterations
        for i in xrange(iterations[c]):
            mask=otsuGray(img[:,:,c],mask)
        
        print mask
       
        if maskinvert[c] == 1:
            output=cv2.bitwise_and(output,cv2.bitwise_not(mask))
        else:
            output=cv2.bitwise_and(output, mask)
    
    return output
     

    
#Applying otsu's algorithm of a grayscale image
def otsuGray(img, mask=None):
    
    hist=np.zeros((256,1))
    
    #Total number of pixels in the image    
    npixels=0.
    mugray=0.;
    threshold=-1;
    maxsigmab=-1;
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if mask is None or mask[i][j] !=0:
                npixels=npixels+1;
                mugray=mugray+img[i,j]
                hist[img[i,j]] = hist[img[i,j]] + 1
            
    # The average grayscale for the entire image
    mugray=mugray/npixels;
    
    #The cumulative probability of pixels less than equal to k
    wi=0
    
    #The cumulative average grayscale value of pixels less than equal to k
    mui=0
    
    
    
    for i in range(0, 256):
        
        #number of pixels at a particular graylevel
        ni=hist[i]
        
        #probability of pixels at level i
        pi=ni/npixels
        
        #Update the cumulative probability of pixels  and also the cumulative average grayscale
        wi=wi+pi
        mui=mui+i*pi
        
        #To avoid the intial and final case this exception is provided so that the 
        #between class sigma does not blow up
        if wi==0 or wi==1:
            continue
        
        
        
        #The between class variance if the threshold was i
        sigmabi=((mui*wi - mui)**2)/(wi*(1-wi))
        
        if sigmabi>maxsigmab:
            threshold=i
            maxsigmab=sigmabi
            
    output=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
        
        #In case the whole whole image was black
    if threshold ==-1:
        return output
            
        #Creating the output
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] > threshold:
                output[i,j]=255
                
    return output
    
#Texture based image representation is returned
def textureImage(img):
    
    #The image is converted to gray scale
    img1=img
    imgg=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    #Initializing the texture based representation
    output=np.zeros_like(img)
    
    #Window sizes used for texture based representation
    wsize=[3,5,7]
    
    for k,w in enumerate(wsize):
        d=w/2
        
        for i in xrange(d, imgg.shape[0]-d):
            for j in xrange(d, imgg.shape[1]-d):
                output[i,j,k]=np.int(np.var(imgg[i-d:i+d+1,j-d:j-d+1]))
                
    return output
                

    
#The contour extraction algorithm. This is done after segmentation  
def contourExtract(img):
    
    #The contour image output is initialised
    output=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
    
    #Do the contouring for each pixel in an image
    for i in xrange(1, img.shape[0]-1):
        for j in xrange(1, img.shape[1]-1):
            if img[i,j]!=0 and np.min(img[i-1:i+2,j-1:j+2])==0:
                output[i,j]=255
            
        
    return output
    

###########################################
# Main method starts here
###########################################

if __name__ == "__main__":
    
    img=cv2.imread('lake.jpg')
    
    texture=0; #Flag to select whether we use texture based image segmentation
    
    if texture==0:
        output=otsuRGB(img,[0,1,1],[1,4,3])
    else:
        outputi=textureImage(img)
        output=otsuRGB(outputi,[1,1,1],[1,1,1])
    
    cv2.imwrite('lake_segment_withnoise.jpg', output)
    
    #Steps to remove noise in the foreground
    kern=np.ones((7,7),np.uint8)
    output=cv2.dilate(output,kern)
    output=cv2.erode(output,kern)   
    
    
    #Steps to remove noise from background
    kern=np.ones((13,13),np.uint8)
    output=cv2.erode(output,kern)
    output=cv2.dilate(output,kern)    
    
    
    
    cv2.imwrite('lake_segment_withoutnoise.jpg', output)
    
    #Extracting the contour
    outputcontour=contourExtract(output)
    
    cv2.imwrite('lake_contour.jpg', outputcontour)
    
    
    
