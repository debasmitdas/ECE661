# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 23:19:15 2016

@author: debasmit
"""

import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

def p_cloud(d_image,K):
    # Inverting K so as P(U)=D(u)*K^-1*U
    K_inv=np.linalg.inv(K)
    point_cloud=[]
    # finding point clouds
    
    for loop1 in range(d_image.shape[0]):
        for loop2 in range(d_image.shape[1]):
            p_cl_tmp=d_image[loop1,loop2]*(np.dot(K_inv,[loop1,loop2,1]))
            
            point_cloud.append(p_cl_tmp)
#    mpl.show()
    return np.asarray(point_cloud)

#########################################
# Point cloud function ends here
#########################################

#########################################
# Display the scatter plot of Images
#########################################

def display(p_cl1,p_cl2,color1,color2,img_name):
    fig = mpl.figure()
    ax = fig.add_subplot(111, projection='3d')
    for loop in range(len(p_cl1)):
        ax.scatter(p_cl1[loop,0],p_cl1[loop,1],p_cl1[loop,2],c=color1)
#    mpl.show()
    for loop in range(len(p_cl2)):
        ax.scatter(p_cl2[loop,0],p_cl2[loop,1],p_cl2[loop,2],c=color2)
#    mpl.show()
    mpl.savefig('point_cloud'+img_name+'.png')
    mpl.close('all')
    return
    
#########################################
# Finding correspondence starts here
#########################################

def Corres(p_cl1,p_cl2,threshold):
    euc_mat=[]
    corres_p=[]
    corres_q=[]
    for loop1 in range(len(p_cl1)):
        for loop2 in range(len(p_cl2)):
            tmp=np.sqrt(sum((p_cl1[loop1,:]-p_cl1[loop2,:])**2))
            euc_mat.append(tmp)
        if np.min(euc_mat)<threshold:
            idx=np.argmin(euc_mat)
            corres_p.append(p_cl1[loop1])
            corres_q.append(p_cl2[idx])
            # to eliminate many to one correspondence
            p_cl2[idx]=np.array([15000,15000,15000])
    
    return np.asarray(corres_p),np.asarray(corres_q)
#########################################
# Finding correspondence ends here
#########################################            
    

#########################################
# ICP algorithm starts here
#########################################
def icp_alg(p_cl1,p_cl2,max_iter,threshold):
    
    for iter in range(max_iter):
        
        # Finding the correspondence with euclidean distance
        p_cl1_d,p_cl2_d=Corres(p_cl1,p_cl2,threshold)
        # Now calculate the centroid for each image
        N=len(p_cl1_d)
        p_centroid=np.array([sum(p_cl1_d[:,0]),sum(p_cl1_d[:,1]),sum(p_cl1_d[:,2])])/(N**2)
        q_centroid=np.array([sum(p_cl2_d[:,0]),sum(p_cl2_d[:,1]),sum(p_cl2_d[:,2])])/(N**2)
        # Now calculate resulting point clouds
        Mp=np.subtract(p_cl1_d,p_centroid)
        Mq=np.subtract(p_cl2_d,q_centroid)
        C=np.dot(Mq.T,Mp)
        # to find the rotation and translation between the images
        U,sig,Vt=np.linalg.svd(C)
        Rot_mat_tmp=np.dot(U,Vt)
        Rot_mat=Rot_mat_tmp.T
        trans_vec=p_centroid-np.dot(Rot_mat,q_centroid)
        tfm_mat=np.zeros((4,4),dtype='float')
        tfm_mat[0:3,0:3]=Rot_mat
        tfm_mat[0:3,3]=trans_vec
        tfm_mat[3,3]=1
        # finding new Q w.r.to transformation matrix 'tfm_mat'
        p_cl2=[]
        for loop in len(p_cl2):
            tmp=np.dot(tfm_mat,[p_cl2[loop,0],p_cl2[loop,1],p_cl2[loop,2],1])
            p_cl2_tmp=tmp[0:3]/float(tmp[3])
            p_cl2.append(p_cl2_tmp)
        p_cl2=np.asarray(p_cl2)
    
    # Final Q is given by i.e transformed point cloud 2 is given by
    Q_trans=p_cl2
    return Q_trans
#########################################
# ICP algorithm ends here
#########################################





###########################################
# Main method starts here
###########################################

if __name__ == "__main__":
    
    img_name1="depthImage1ForHW.txt"
    img_name2="depthImage2ForHW.txt"
    max_iter=20;
    K=np.array([[365,0,256],[0,365,212],[0,0,1]]);
    threshold=0.1
    d_image1=np.loadtxt(img_name1)
    d_image2=np.loadtxt(img_name2)
    img_name1=img_name1.split('.')[0]
    img_name2=img_name2.split('.')[0]

    
    mpl.imshow(d_image1,origin='upper',extent=[0, d_image1.shape[1], 0, d_image1.shape[0]])
    mpl.colorbar()
#    mpl.show()
    mpl.savefig('op_'+img_name1+'.png')
    mpl.close()
    mpl.imshow(d_image2,origin='upper',extent=[0, d_image2.shape[1], 0, d_image2.shape[0]])
    mpl.colorbar()
#    mpl.show()
    mpl.savefig('op_'+img_name2+'.png')
    mpl.close()
    
    
    point_cloud_img1=p_cloud(d_image1,K)
    point_cloud_img2=p_cloud(d_image2,K)
    img_name="_before_ICP"
    print "The output is saved in point_cloud"+img_name+".png"
    display(point_cloud_img1,point_cloud_img2,'b','g',img_name)
    
    pt_cl_2_trans=icp_alg(point_cloud_img1,point_cloud_img1,max_iter,threshold)
    
    img_name="_after_ICP"
    print "The output is saved in point_cloud"+img_name+".png"
    display(point_cloud_img1,pt_cl_2_trans,'b','r',img_name)
    
    
    