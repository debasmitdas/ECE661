function [im1rect im2rect F x1new x2new H1 H2] = Rect_Img(e1,e2,x1,x2,im1,im2,P1,P2,F)

h=size(im1,1);
w=size(im1,2);
npts=size(x1,1);

%Convert from homogeneous to physical coordinates
e2=e2/e2(end);

ang=atan(-(e2(2)-h/2)/(e2(1)-w/2));
f=cos(ang)*(e2(1)-w/2)-sin(ang)*(e2(2)-h/2);
R=[cos(ang) -sin(ang) 0;sin(ang) cos(ang) 0; 0 0 1];
T=[1 0 -w/2;0 1 -h/2;0 0 1];
G=[1 0 0;0 1 0;-1/f 0 1];
H2=G*R*T;

%Preserve Centre after applying Homography
cpt=[w/2 h/2 1]';
ncpt=H2*cpt;
ncpt=ncpt/ncpt(end);

T2=[1 0 w/2-ncpt(1);0 1 h/2-ncpt(2);0 0 1];
H2=T2*H2;



% H1 computation 
%Compute homography for first image 
M=P2*pinv(P1);
H0=H2*M;
H0=H1;
x1hat=ones(size(x1));
x2hat=ones(size(x2));

for i=1:1:npts
    tmp=(H0*x1(i,:)')'
    x1hat(i,:)=tmp/tmp(end);
    tmp=(H2*x2(i,:)')';
    x2hat(i,:)=tmp/tmp(end);
end

% %Linear Least Squares for finding HA
A=zeros(npts,3);
b=zeros(npts,1);
for i=1:npts
    A(i,:)=[x1hat(i,1) x1hat(i,2) 1];
    b(i)= x2hat(i,1);
end
x=pinv(A)*b; % Least squares step
HA=[x(1) x(2) x(3); 0 1 0; 0 0 1];
H1=HA*H0;

cpt=[w/2 h/2 1]';
ncpt=H1*cpt;
ncpt=ncpt/ncpt(end);

T1=[1 0 w/2-ncpt(1);0 1 h/2-ncpt(2);0 0 1];
H1=T1*H1;

% Update the Fundamental Matrix 

[im1rect H1]=appHomo(H2,im1);
[im2rect H2]=appHomo(H2,im2);
F=inv(H2')*F*inv(H1);

% Update the interest points
x1new =zeros(size(x1));
x2new=zeros(size(x2));

for i=1:npts
    tmp=(H1*x1(i,:)')';
    x1new(i,:)=tmp/tmp(end);
    tmp=(H2*x2(i,:)')';
    x2new(i,:)=tmp/tmp(end);
end
end












