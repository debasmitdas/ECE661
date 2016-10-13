"""
Author : Debasmit Das
"""

#import cv2
import numpy as np
import math

####################################
#Algorithm for RANSAC starts here
####################################
def Ransac(pts_src,pts_dest,gauss_noise,error,n,p_no_outlier):
    H=np.zeros((3,3),dtype='float')
    pos_sol=[]
    # Spatial resolution is not considered for this case to set the threshold delta
    delta=float(3*gauss_noise)
    N=int(math.log(1-p_no_outlier)/math.log(1-(1-error)**n))
    M=int((1-error)*len(pts_src))
    print "The no. of Iteration %d and Iter Max count are %d" %(N,M)
    for loop in range(N):
        random_index=np.random.random_integers(1,len(pts_src)-1,n)
        pts_1=[]
        pts_2=[]
        for loop in range(n):
            pts_1.append(pts_src[random_index[loop],:])
            pts_2.append(pts_dest[random_index[loop],:])
        pts_tmp1=np.asarray(pts_1)
        pts_tmp2=np.asarray(pts_2)
        # If dHomography function is used it uses the solution of Ah=b
        # While Homography function uses solution to Ah=0
        H=Homography(pts_tmp1,pts_tmp2)
        InlierCount= Inliers_count(pts_src,H,pts_dest,delta)
        H_vec=np.reshape(H,9)
        
        if InlierCount > M:
            pos_sol.append([H_vec,InlierCount])
        
    # Finally obtaining the Final homography
    solution=np.asarray(pos_sol)
    print solution
    idx_MaxInlierCount = np.argmax(solution[:,-1])
    #print len(solution)
    
    H_ransac=np.reshape(solution[idx_MaxInlierCount,0],(3,3))
    #H[1]=solution[idx_MaxInlierCount,0:3]
    #H[2]=solution[idx_MaxInlierCount,3:6]
    #H[3]=solution[idx_MaxInlierCount,6:9]
    return H_ransac

#####################################
# Finding Homography
####################################

def Homography(points_src,points_dest):
    Mat_A=np.zeros((len(points_src)*2,9),dtype='float')
    #b = np.zeros((1,8))
    if points_src.shape[0] != points_dest.shape[0] or points_src.shape[1] != points_dest.shape[1]:
        print "No. of Source and destination points donot match"
        exit(1)
    
    for i in range(0,len(points_src)):
        Mat_A[i*2+1]=[points_src[i,0],points_src[i,1],1,0,0,0,(-1*points_src[i,0]*points_dest[i,0]),(-1*points_src[i,1]*points_dest[i,0]),-points_dest[i,0]]
        Mat_A[i*2]=[0,0,0,-points_src[i,0],-points_src[i,1],-1,(1*points_src[i,0]*points_dest[i,1]),(1*points_src[i,1]*points_dest[i,1]),points_dest[i,1]]
               
        #b[0][i*2] = points_dest[i][0]
        #b[0][i*2+1] = points_dest[i][1]        
    # A matrix formed to solve Ah=0
      
    # If no. of points is not 4 then using the below code
    #tmp_H=np.dot(np.linalg.pinv(Mat_A),b.T)
    U,D,V=np.linalg.svd(Mat_A)
    V_tmp=V.T
    tmp_H=V_tmp[:,-1]
    homography= np.zeros((3,3))
    homography[0]= tmp_H[0:3]/tmp_H[8]
    homography[1]= tmp_H[3:6]/tmp_H[8]
    homography[2]= tmp_H[6:9]/tmp_H[8]
    #print homography
    return homography
####################################
# Function for finding Homography ends here
####################################
    
#####################################
# Finding Inlier Count
####################################
def Inliers_count(pts_src,H,pts_dest,delta):
    result=np.zeros((pts_src.shape),dtype='int')
    
    for loop in range(len(pts_src)):
        result_tmp=np.dot(H,([pts_src[loop,0],pts_src[loop,1],1]))
        result[loop,0]=result_tmp[0]/result_tmp[2]
        result[loop,1]=result_tmp[1]/result_tmp[2]
    diff=result-pts_dest
    #print diff
    sq_dist=np.square(diff)
    Dist=np.sqrt(sq_dist[:,0]+sq_dist[:,1])
    #print Dist
    Inliers=[1 for val in Dist if (val < delta)]
    Inlier_count= len(Inliers)
    return Inlier_count
#####################################
# Finding Inlier Count ends here
####################################

#####################################
# Finding Inlier for Finalized homography starts here 
####################################
def Inliers(pts_src,pts_dest,H,delta):
    result=np.zeros((pts_src.shape),dtype='int')
    
    for loop in range(len(pts_src)):
        result_tmp=np.dot(H,([pts_src[loop,0],pts_src[loop,1],1]))
        result[loop,0]=result_tmp[0]/result_tmp[2]
        result[loop,1]=result_tmp[1]/result_tmp[2]
    diff=result-pts_dest
    #print diff
    sq_dist=np.square(diff)
    Dist=np.sqrt(sq_dist[:,0]+sq_dist[:,1])
    
    Inlier=[]
    for loop in range(len(Dist)):
        if Dist[loop] < delta:
            Inlier.append([pts_src[loop,:],pts_dest[loop,:]])
    
    return np.asarray(Inlier)
#####################################
# Finding Inlier for Finalized homography ends here
####################################
    
#####################################
# Finding Homography Using Ah=b method
####################################

def dHomography(points_src,points_dest):
    Mat_A=np.zeros((len(points_src)*2,8),dtype='float')
    b = np.zeros((1,len(points_src)*2))
    if points_src.shape[0] != points_dest.shape[0] or points_src.shape[1] != points_dest.shape[1]:
        print "No. of Source and destination points donot match"
        exit(1)
    
    for i in range(0,len(points_src)):
        Mat_A[i*2]=[points_src[i,0],points_src[i,1],1,0,0,0,(-1*points_src[i,0]*points_dest[i,0]),(-1*points_src[i,1]*points_dest[i,0])]
        Mat_A[i*2+1]=[0,0,0,points_src[i,0],points_src[i,1],1,(-1*points_src[i,0]*points_dest[i,1]),(-1*points_src[i,1]*points_dest[i,1])]
               
        b[0][i*2] = points_dest[i][0]
        b[0][i*2+1] = points_dest[i][1]        
    # A matrix formed to solve Ah=0
      
    # If no. of points is not 4 then using the below code
    tmp_H=np.dot(np.linalg.pinv(Mat_A),b.T)
    #U,D,V=np.linalg.svd(Mat_A)
    #print V[:,-1]
    #tmp_H=V[:,-1]
    homography= np.zeros((3,3))
    homography[0]= tmp_H[0:3].T
    homography[1]= tmp_H[3:6].T
    homography[2,0:2]= tmp_H[6:8].T
    homography[2,2]=1
    return homography
####################################
# Function for finding Homography ends here
####################################