%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; warning off;
%Now we have to initialize the world co-ordinates measured with ruler
xW=zeros(80,2)
for j=1:8
    for i=1:10
        xW((j-1)*10+i,:)=[(j-1)*25 (i-1)*25];
    end
end

nimg=20; % The number of images used
rad_dist=1; % This is the indicator of whether radial distortion is used.

HAll=[];V=[];xIM=[];
for k =1:20
 filename = strcat('Dataset2/Pic_',int2str(k),'.jpg');
 %the coordinates of the corners in the image
 [imcoord] = findCorners(filename);
 xIM{k} = imcoord;
 %solve Ah = 0, and find A for the homography
 A = findA(xW(:,1),xW(:,2),double(imcoord(:,1)),double(imcoord(:,2)));
 [U,D,T] = svd(A);
 h = T(:,9);
 H = [h(1:3)'; h(4:6)'; h(7:9)'];
 HAll{k} = H;
 %V matrix for calculating the intrinsic parameters
 i=1;j=2;
 v12 = [H(1,i)*H(1,j), H(1,i)*H(2,j)+H(2,i)*H(1,j), H(2,i)*H(2,j), H(3,i)*H(1,j)+H(1,i)*H(3,j) ,H(3,i)*H(2,j)+H(2,i)*H(3,j),H(3,i)*H(3,j)];
 i=1;j=1;
 v11 = [H(1,i)*H(1,j), H(1,i)*H(2,j)+H(2,i)*H(1,j), H(2,i)*H(2,j), H(3,i)*H(1,j)+H(1,i)*H(3,j) ,H(3,i)*H(2,j)+H(2,i)*H(3,j),H(3,i)*H(3,j)];
 i=2;j=2;
 v22 = [H(1,i)*H(1,j), H(1,i)*H(2,j)+H(2,i)*H(1,j), H(2,i)*H(2,j), H(3,i)*H(1,j)+H(1,i)*H(3,j) ,H(3,i)*H(2,j)+H(2,i)*H(3,j),H(3,i)*H(3,j)];
 V = [V;v12;(v11-v22)];
end

[U,D,T] = svd(V);
b = T(:,6); %B11 B12 B22 B13 B23 B33
%intrinsic parameters
y0 = (b(2)*b(4)-b(1)*b(5))/(b(1)*b(3)-b(2)^2);
lambda = b(6)-(b(4)^2+y0*(b(2)*b(4)-b(1)*b(5)))/b(1);
ax = sqrt(lambda/b(1));
ay = sqrt(lambda*b(1)/(b(1)*b(3)-b(2)^2));
s = -b(2)*ax^2*ay/lambda;
x0 = s*y0/ay-b(4)*ax^2/lambda;
K = [ax s x0; 0 ay y0; 0 0 1];

p = zeros(1,5+6*nimg);
p(1:5) = [ax s x0 ay y0];
if(rad_dist)
 p = zeros(1,7+6*nimg);
 p(1:5) = [ax s x0 ay y0];
 p(6:7) = [0 0];
 cnt = 7;
else
 p = zeros(1,5+6*nimg);
 p(1:5) = [ax s x0 ay y0];
 cnt = 5;
end

ydata=[];
K_inv = inv(K);
R_b4LM = [];
R_LM = [];
t_b4LM = [];
t_LM = [];

%This is for intrinsic parameters R,t
%The R is also converted to Rodriguez formula
%for w and theta
for k = 1:nimg
 H = HAll{k};
 t = K_inv*H(:,3);
 mag = norm(K_inv*H(:,1));
 if(t(3)<0)
 mag = -mag;
 end
 r1 = K_inv*H(:,1)/mag;
 r2 = K_inv*H(:,2)/mag;
 r3 = cross(r1,r2);
 R = [r1 r2 r3];
 t = t/mag;
 [U,D,V] = svd(R);
 R = U*V';
 R_b4LM{k}=R;
 t_b4LM{k}=t;
 % Rodriguez formula used here
 phi = acos((trace(R)-1)/2);
 w = phi/(2*sin(phi))*([R(3,2)-R(2,3) R(1,3)-R(3,1) R(2,1)-R(1,2)])';
 p(cnt+1:cnt+3) = w;
 p(cnt+4:cnt+6) = t;
 cnt = cnt + 6;
 y=xIM{k};
 y=y';
 ydata=[ydata y(:)'];
end
x = xW';
xdata = x(:)';

% LM algorithm is carried out for refinement
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
p1 = lsqnonlin(@dgeom,p,[],[],options,xdata,ydata,rad_dist,nimg);

% Finding the intrinsic calibration matrix
ax = p1(1);
s = p1(2);
x0 = p1(3);
ay = p1(4);
y0 = p1(5);
K1 = [ax s x0; 0 ay y0; 0 0 1];
if(rad_dist)
 k1 = p1(6);
 k2 = p1(7); % Finding the radial distortion parameters
 cnt = 7;
else
 cnt = 5;
end

%Converting back to R and t for extrinsic parameters after LM
for k = 1:nimg
 w = p1(cnt+1:cnt+3);
 t_LM{k} = p1(cnt+4:cnt+6)';
 cnt = cnt + 6;
 wx = [0 -w(3) w(2); w(3) 0 -w(1); -w(2) w(1) 0];
 phi = norm(w);
 R_LM{k} = eye(3)+sin(phi)/phi*wx + (1-cos(phi))/phi*wx^2;
end







