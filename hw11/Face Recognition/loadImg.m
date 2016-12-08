function [ normImgVec, imgVec, meanImg ] = loadImg( filePath, Npers, Ntrial )
%loadImages Summary of this function goes here
% Detailed explanation goes here
%get image size
img = imread([filePath,'01_01.png']);
imgGray = rgb2gray(img);
[row,col] = size(imgGray);
%define output vectors
imgVec = zeros(row*col,Npers*Ntrial); %each column is an image
%load images into 1D vectors
for i = 1:Npers
 for j = 1:Ntrial
 img = imread([filePath,num2str2digit(i),'_',num2str2digit(j),'.png']);
 %figure;
 %imshow(img);
 imgGray = rgb2gray(img);
 [row,col] = size(imgGray);
 oneVec = reshape(imgGray',row*col,1);
 imgVec(:,(i-1)*Ntrial+j) = oneVec;
 end
end
%compute mean of all images
meanImg = mean(imgVec,2);
%normalize images using the mean
normImgVec = zeros(row*col,Npers*Ntrial);
for i = 1:Npers*Ntrial
 normImgVec(:,i) = (imgVec(:,i) - meanImg) / norm(imgVec(:,i) - meanImg);
end
end