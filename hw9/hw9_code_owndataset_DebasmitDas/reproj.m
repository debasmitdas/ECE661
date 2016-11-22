function [] = reproj(R,t,K,xW,k)
% R is the list of Calibrated rotation matrices
% t is the list of calibrated translation vectors
% K is the intrinsic camera calibration matrix
% xW is the list of corner world coordinates
% k is the index  of the projected image
%fixed image is image 4 for our dataset
filename = strcat('Dataset2/Pic_',int2str(4),'.jpg');
img = rgb2gray(imread(filename));
Pfix = K*[R{4}(:,1:2) t{4}]; %This is the Homography for the fixed image
xtrue = xW{4};
P = K*[R{k}(:,1:2) t{k}];%This is the Homography for the projected image
x0 = K(1,3);
y0 = K(2,3); % These are the co-ordinates of the principal point 
xi = xW{k};
xi = [xi ones(size(xi,1),1)];
xyz = inv(P)*xi';
xest = (Pfix*xyz)';
figure
imshow(img)
%Now plotting the reprojections
for i = 1:80
 xest(i,:) = xest(i,:) / xest(i,3);
 hold on
 plot(uint64(xtrue(i,1)),uint64(xtrue(i,2)),'g.','MarkerSize',12);
 hold on
 plot(uint64(xest(i,1)),uint64(xest(i,2)),'r.','MarkerSize',12);
end
xest = xest(:,1:2);
hold off
mean(abs(xtrue(:)-xest(:))); %Plotting the mean of error
var(abs(xtrue(:)-xest(:))); %Plotting the variance of error
end