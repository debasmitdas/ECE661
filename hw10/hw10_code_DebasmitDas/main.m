close all; clear; warning off

im1=imread('left.png');
im2=imread('right.png');

% Downsampling is down if the resolution is too high
fact=1
im1 = imresize(im1, fact);
im2 = imresize(im2, fact);

% Parameters
n_corresp=8; % Total no. of manual correspondences
T_ncc=0.7; % Threshold for NCC
r_ncc=0.99; % Ratio for getting rid of false correspondences
thresh_F=1e-16; % Threshold for numerical purposes since we may not get
% x'^TFx

%% Selecting the manual correspondences
pt1=[124 234 ; 124 208 ; 232 124 ; 376 264 ; 370 304 ; 262 398 ; 264 368 ; 257 252];
pt2=[135 218 ; 136 192 ; 252 126 ; 369 282 ; 364 320 ; 238 396 ; 241 366 ; 255 254];


% Making them homogeneous coordinates
x1=[pt1 ones(n_corresp,1)];
x2=[pt2 ones(n_corresp,1)];

% Plot the correspondences
plot_corresp(im1,im2,x1,x2);


%% Finding F

%Use the 8-point normalization algorithm
[T1]= norm_point(x1);
[T2]= norm_point(x2);

%Find F using Linear Least Squares
F=Find_F_LLS(x1,x2,T1,T2);

% 

%Compute the epipoles and the projection  matrices
e1=null(F); % Right null vector
e2=null(F'); % Left null vector

e2x = [0 -e2(3) e2(2);e2(3) 0 -e2(1);-e2(2) e2(1) 0];


P1=[1 0 0 0;0 1 0 0;0 0 1 0];
P2=[e2x*F, e2];

% Apply Non-linear least  squares to imporve estimate of F and P2;
[P2 F X]=NLLSOpt(@errorFunc,x1,x2,P1,P2);


%Recalculate the epipoles with refined estimates
e1=null(F);
e2=null(F');

%% Rectification to be done

%Rectification of images need to be done
[im1rect im2rect F x1new x2new H1 H2] = Rect_Img(e1,e2,x1,x2,im1,im2,P1,P2,F);


plot_corresp(im1rect,im2rect,x1new,x2new);

%% Extracting SURF Features

Ig1=rgb2gray(im1rect);
Ig2=rgb2gray(im2rect);

pts1=detectSURFFeatures(Ig1);
pts2=detectSURFFeatures(Ig1);

[feat1,valpts1]= extractFeatures(Ig1,pts1);
[feat2,valpts2]= extractFeatures(Ig2,pts2);

% This is for finding the NCC matrix
[C]=EstCorrespNCC(feat1,feat2,valpts1,valpts2);
[x1f,x2f]=GetFinFeatNCC(C,feat1,feat2,valpts1,valpts2, r_ncc, T_ncc, F, thresh_F );

% Plot the Correspondences for SURF
plot_corresp(im1rect,im2rect,x1f,x2f);

%% Beginning of 3D reconstruction 

[T1]= norm_point(x1f);
[T2]= norm_point(x2f);

%Find F using Linear Least Squares
F=Find_F_LLS(x1f,x2f,T1,T2);

%Compute the epipoles and the projection  matrices
e1=null(F); % Right null vector
e2=null(F'); % Left null vector

e2x = [0 -e2(3) e2(2);e2(3) 0 -e2(1);-e2(2) e2(1) 0];

%Compute Camera Projection Matrices 
P1=[1 0 0 0;0 1 0 0;0 0 1 0];
P2=[e2x*F, e2];

% Apply Non-linear least  squares to imporve estimate of F and P2;
[P2 F X]=NLLSOpt(@errorFunc,x1f,x2f,P1,P2);

PlotWorPts(X,x1f,x2f,P1,P2,F,thresh_F);





