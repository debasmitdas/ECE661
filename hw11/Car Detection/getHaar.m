function [feats, Npos, Nneg] = getHaar(filePath)
% This is is for getting the features from all images

%The size of the images are fixed
r=20; % The no. of rows
c=40; % The no. of columns

%Setting the File paths 
posFilePath = [filePath 'positive/'];
negFilePath = [filePath 'negative/'];
disp(posFilePath);
posImg = loadImagesAdaboost(posFilePath, r, c);
negImg = loadImagesAdaboost(negFilePath, r, c);

% get total number of images
Nimg = size(posImg,3) + size(negImg,3);
Npos = size(posImg,3);
Nneg = size(negImg,3);

Nfeats=166000;
feats=zeros(Nfeats, Nimg);

for i=1:Nimg
    intImg=zeros(r+1,c+1);
    disp(i);
    if i<=size(posImg,3)
        intImg(2:r+1,2:c+1) = cumsum(cumsum(posImg(:,:,i)),2);
    else
        intImg(2:r+1,2:c+1) = cumsum(cumsum(negImg(:,:,i-size(posImg,3))),2);
    end
    feats(:,i)=computeFeature(intImg);
end

features_adaboost.features = feats;
features_adaboost.Npos = Npos;
features_adaboost.Nneg = Nneg;
% For saving test image features
save('features_adaboost_test.mat', 'features_adaboost', '-mat', '-v7.3');
% For saving training image features
%  save('features_adaboost_train.mat', 'features_adaboost', '-mat', '-v7.3');
end
        