function [ vecU, Z ] = myLDA( imgVec, mean, Nperson, Ntrials )
%myLDA Summary of this function goes here
% Detailed explanation goes here
% define image size
imgSize = 128*128;

% compute mean for each class
sumImg = zeros(imgSize,Nperson*Ntrials);
for i = 1:Nperson*Ntrials
 classIdx = floor((i-1)/Ntrials) + 1;
 sumImg(:,classIdx) = sumImg(:,classIdx) + imgVec(:,i);
end
meani = sumImg / Ntrials;
% build mi-m
meani_m = zeros(imgSize, Nperson);
for i = 1:Nperson
 meani_m(:,i) = meani(:,i) - mean;
end
% compute SB
SB = meani_m * meani_m';
% ensure SB is not singular
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
for i = 1:Nperson*Ntrials
 classIdx = floor((i-1)/Ntrials) + 1;
 xk_meani(:,i) = imgVec(:,i) - meani(:,classIdx);
end
% compute Zt*Sw*Z = Z' * (xk-meani) * (xk-meani)' * Z
Zt_xk_meani = Z' * xk_meani;
% eigendecompostion to get U
[vecU,valU] = eig(Zt_xk_meani*Zt_xk_meani');
% diagnolize eigenvalues of U
DU = diag(valU);
end