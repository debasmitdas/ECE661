function h = getClassifier(features, weight, label, Npos)
% This function is used for getting the classifier
% To define parameters
Nfeatures = size(features,1);
Nimgs = size(features,2);
h.currentMin = inf;

tPos = repmat(sum(weight(1:Npos,1)), Nimgs,1);
tNeg = repmat(sum(weight(Npos+1:Nimgs,1)), Nimgs,1);


% search each feature as a classifier
for i = 1: Nfeatures
% get one feature for all images
oneFeature = features(i,:);
% sort feature to thresh for postive and negative
[sortedFeature, sortedIdx] = sort(oneFeature, 'ascend');
% sort weights and labels
sortedWeight = weight(sortedIdx); sortedLabel = label(sortedIdx);
  % select threshold
 sPos = cumsum(sortedWeight .* sortedLabel);
 sNeg = cumsum(sortedWeight) - sPos;
 errPos = sPos + (tNeg - sNeg);
 errNeg = sNeg + (tPos - sPos);

 % choose the threshold with small error
 allErrMin = min(errPos, errNeg);
 [errMin, idxMin] = min(allErrMin);

 % result
 result = zeros(Nimgs,1);
 if errPos(idxMin) <= errNeg(idxMin)
 p = -1; % Setting the polarity to negative
 result(idxMin+1:end) = 1;
 result(sortedIdx) = result;
 else
 p = 1; % Setting the polarity to positive
 result(1:idxMin) = 1;
 result(sortedIdx) = result;
 end

 % get best parameters
 if errMin < h.currentMin
 h.currentMin = errMin;
 if idxMin==1
 h.theta = sortedFeature(1) - 0.5;
 elseif idxMin==Nfeatures;
 h.theta = sortedFeature(Nfeatures) + 0.5;
 else
 h.theta = (sortedFeature(idxMin)+sortedFeature(idxMin-1))/2;
 end

 h.p = p;
 h.featureIdx = i;
 h.bestResult = result;
 end

end % end of search each feature
end
 