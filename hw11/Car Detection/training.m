% This is the script for trainining the object detector

clc
clear all

% Get features from stored data file
featFile = load('features_adaboost_train.mat');
feat=featFile.features_adaboost.features;
Npos=featFile.features_adaboost.Npos;
Nneg=featFile.features_adaboost.Nneg;

S=10;
idx=1:Npos+Nneg;

for i=1:S
    idx=myCascade(feat,Npos,idx,i);
    
    if length(idx)==Npos
        break;
    end
end


