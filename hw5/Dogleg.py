"""
Author: Debasmit Das
"""

import numpy as np

####################################    
# Minimizing Homography's error using Dogleg 
#################################### 
def Dogleg(pts_srcX,pts_destX1,H):
    # pass the homography such that it maps X1 to X
    
    #Initialize r=1
    r=1
    T=0.6 # Setting T to any value between 0 and 1
    # compute Jacobian of the homography function
    Jf=Jacobian_fn(pts_destX1,H)
    # Computing u
    u=T*np.max(np.diag(np.dot(Jf.T,Jf)))
    #u=0.000005
    pts_Xlist=[]
    for loop in range(len(pts_srcX)):
        pts_Xlist.append(pts_srcX[loop,0])
        pts_Xlist.append(pts_srcX[loop,1])
    pts_X=np.asarray(pts_Xlist)
    while True:    
    # Finding f(p)    
        fp=Apply_Homography(pts_destX1,H)
        # compute episilon pk
        ep=pts_X-fp
        # compute the cost function
        Cp=(np.linalg.norm(ep)**2)
        Jf=Jacobian_fn(pts_destX1,H)
        # compute delta for gradient descent
        Jt_ep_k= np.dot(Jf.T,ep)
        
        delta_GD=np.linalg.norm(Jt_ep_k)*Jt_ep_k/np.linalg.norm(np.dot(Jf,Jt_ep_k))
        # compute delta for Gauss-Newton
        delta_GN=np.dot(np.linalg.inv(np.dot(Jf.T,Jf)+u*np.identity(9)),Jt_ep_k)
        delta_p=np.linalg.inv(np.dot(Jf.T,Jf)).dot(np.dot(Jf.T,ep))
        H_old=H
        
        if np.linalg.norm(delta_GN)<r:
            H=H+np.reshape(delta_GN,(3,3))
        elif np.linalg.norm(delta_GN)>= r and np.linalg.norm(delta_GD)<=r:
        
            b=2*(delta_GD.T).dot(delta_GN-delta_GD)
            F_ac=(b**2)-4*(np.linalg.norm(delta_GN-delta_GD)**2)*(np.linalg.norm(delta_GD)**2-r**2)
            beta=(-b+np.sqrt(F_ac))/(2*np.linalg.norm(delta_GN-delta_GD)**2)
            #beta_eqn=[np.linalg.norm(delta_GN-delta_GD)**2,2*(delta_GD.T).dot(delta_GN-delta_GD),(np.linalg.norm(delta_GD)**2-r**2)]
            #beta1=np.roots(beta_eqn)
            H=H+np.reshape(delta_GD,(3,3))+beta*(np.reshape(delta_GN,(3,3))-np.reshape(delta_GD,(3,3)))
        else:
            H=H+(r/np.linalg.norm(delta_GD))*np.reshape(delta_GD,(3,3))
        fp1=Apply_Homography(pts_destX1,H)
        Cp1=(np.linalg.norm(pts_X-fp1)**2)
        delta_DL=(Cp-Cp1)/(2*delta_p.dot(Jt_ep_k)-np.dot(np.dot(Jf,delta_p).T,np.dot(Jf,delta_p)))
        #print delta_DL
        u=u*max(1/3,(1-(2*delta_DL-1)**3))
        if delta_DL<=0:
            H=H_old
            r=r/2
        elif delta_DL<0.25:
            r=r/4
        elif delta_DL<0.75:
            r=r 
            # do nothing
        else:
            r=2*r
        if Cp-Cp1 < 0.001:
            break
        else:
            continue
        # 0.001 is the threshold value in which the loop stops
    return H
####################################    
# Dogleg function ends here
####################################   

####################################    
# Applying homography to points starts here
#################################### 

def Apply_Homography(pts_src,H):
    result=np.zeros((pts_src.shape),dtype='float')
    output=[]
    for loop in range(len(pts_src)):
        result_tmp=np.dot(H,([pts_src[loop,0],pts_src[loop,1],1]))
        result[loop,0]=result_tmp[0]/result_tmp[2]
        result[loop,1]=result_tmp[1]/result_tmp[2]
        output.append(result[loop,0])
        output.append(result[loop,1])
    return np.asarray(output)
####################################    
# Applying homography to points ends here
#################################### 

#################################### 
# Finding Jacobian of fx
#################################### 
def Jacobian_fn(pts_X1,H):
    # create a base matrix
    Jacobian=np.zeros((2*len(pts_X1),9),dtype='float')
    for loop in range(len(pts_X1)):
        
        tmp_f=np.dot(H,([pts_X1[loop,0],pts_X1[loop,1],1]))
        Jacobian[loop*2]=[pts_X1[loop,0]/tmp_f[2],pts_X1[loop,1]/tmp_f[2],1/tmp_f[2],0,0,0,-pts_X1[loop,0]*tmp_f[0]/(tmp_f[2]**2),-pts_X1[loop,1]*tmp_f[0]/(tmp_f[2]**2),-1*tmp_f[0]/(tmp_f[2]**2)]
        Jacobian[loop*2+1]=[0,0,0,pts_X1[loop,0]/tmp_f[2],pts_X1[loop,1]/tmp_f[2],1/tmp_f[2],-pts_X1[loop,0]*tmp_f[1]/(tmp_f[2]**2),-pts_X1[loop,1]*tmp_f[1]/(tmp_f[2]**2),-1*tmp_f[1]/(tmp_f[2]**2)]
    return Jacobian