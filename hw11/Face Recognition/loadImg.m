function [ normImgVec, imgVec, meanImg ] = loadImg( filePath, Npers, Ntrial )

%get image dimensions
[r,c] = size(rgb2gray(imread([filePath,'01_01.png'])));

%define output vectors
imgVec = zeros(r*c,Npers*Ntrial); %each column is an image

%load images as feature vectors
for i = 1:Npers
 for j = 1:Ntrial
 img = imread([filePath,num2str2digit(i),'_',num2str2digit(j),'.png']);
 [r,c] = size(rgb2gray(img));
 oneVec = reshape(rgb2gray(img)',r*c,1);
 imgVec(:,(i-1)*Ntrial+j) = oneVec;
 end
end

%compute ensemble mean of all images
meanImg = mean(imgVec,2);

%normalize images
% This standardization is required for eigen decomposition
normImgVec = zeros(r*c, Npers*Ntrial);
for i = 1:Npers*Ntrial
 normImgVec(:,i) = (imgVec(:,i) - meanImg) / norm(imgVec(:,i) - meanImg);
end
end
