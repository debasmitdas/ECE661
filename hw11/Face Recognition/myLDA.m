function [ vecU, Z ] = myLDA( imgVec, mean, Npers, Ntrials )
%myLDA Summary of this function goes here
% Detailed explanation goes here
% define image size
imgSize = 128*128;
% compute mean for each class
sumImg = zeros(imgSize,Npers*Ntrials);
for i = 1:Npers*Ntrials
 % This is for selecting index for each class
 classIdx = floor((i-1)/Ntrials) + 1;
 sumImg(:,classIdx) = sumImg(:,classIdx) + imgVec(:,i);
end

meani = sumImg / Ntrials;

% build mi-m after subtracting from mean
meani_m = zeros(imgSize, Npers);
for i = 1:Npers
 meani_m(:,i) = meani(:,i) - mean;
end

% compute SB i.e the between class variance
SB = meani_m * meani_m';
% ensure SB is not singular using Yu and Wang's method
[vecSB,valSB] = eig(meani_m' * meani_m);
[~,idx] = sort(-1 .* diag(valSB));
V = meani_m * vecSB;


Nfeatures = 30;
% build Y, DB, Z
Y = V(:,1:Nfeatures);
DB = Y' * meani_m * meani_m' * Y;
Z = Y * DB^(-0.5);
% build xk-mi
xk_meani = zeros(imgSize, Ntrials);
for i = 1:Npers*Ntrials
 classIdx = floor((i-1)/Ntrials) + 1;
 xk_meani(:,i) = imgVec(:,i) - meani(:,classIdx);
end

% compute the intermediate variable
Zt_xk_meani = Z' * xk_meani;
% eigendecompostion to get U
[vecU,valU] = eig(Zt_xk_meani*Zt_xk_meani');
% diagnolize eigenvalues of U
DU = diag(valU);

end
