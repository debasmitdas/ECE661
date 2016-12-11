function [ result ] = adaBoostClassify( featuresAll, alpha, p, th, fIdx, T)
% th is the threshold and p is the polarity
%Summary of this function goes here
% Detailed explanation goes here
% get number of test images
Nimgs = size(featuresAll,2);
% result for each weak classifier
weakResult = zeros(Nimgs,T);
% classify using every weak classiier
for t = 1:T
 % get classifier feature
 feature = featuresAll(fIdx(t),:);
 % do classification for each test image
 for i = 1:Nimgs
 if p(t)*feature(i) <= p(t)*th(t)
 weakResult(i,t) = 1; %else it will be negative class
 end
 end
end
% build strong classifier by weighted average of weak classifiers
strongCla = weakResult(:,1:T) * alpha(1:T,:);
% compute strong classifier thershold
strongTh = 0.5 * sum(alpha(1:T,1));
% get final classification result
result = zeros(Nimgs,1);
for i = 1:Nimgs
 if strongCla(i) >= strongTh
 result(i) = 1;
 end
end
end