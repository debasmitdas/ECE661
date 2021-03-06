function [idx] = myCascade(featuresAll, Npos, idxPrev, stage)

%update negative no.
Nneg = length(idxPrev) - Npos;
Ntotal = Npos + Nneg;

%update features
feats = featuresAll(:,idxPrev);

% Initialize weights to equally probability
weight = zeros(Ntotal,1);

% Initialize labels for both positive and negative examples
label = zeros(Ntotal,1);

for i=1:Ntotal
    if i <= Npos
        weight(i) = 0.5 / Npos;
        label(i) = 1;
    else
        weight(i) = 0.5 / Nneg;
    end
end

% This is the adaboost process
T=40;
strongClaResult = zeros(Ntotal, 1);
alpha = zeros(T,1);
ht = zeros(4,T);
hRes = zeros(Ntotal, T);

% The adaboost iterative process 
for t = 1:T
% normalize weights
weight = weight ./ sum(weight);
% get the best weak classifier and the detection result
h = getClassifier(feats, weight, label, Npos);
% store result
ht(1,t) = h.currentMin;
ht(2,t) = h.p;
ht(3,t) = h.featureIdx;
ht(4,t) = h.theta;
hRes(:,t) = h.bestResult;
% get min error
err = h.currentMin;
% get trust fact alphat = 0.5 * ln((1-et)/et)
alpha(t) = log((1-err)/err);

% update weight
weight = weight .* (err/(1-err)) .^ (1-xor(label,h.bestResult));

% strong classifier
strongCla = hRes(:,1:t) * alpha(1:t,:);
threshold = min(strongCla(1:Npos));

for i = 1:Ntotal
if strongCla(i) >= threshold
strongClaResult(i) = 1;
else
strongClaResult(i) = 0;
end
end

% compute positive accuracy
posAccuracy(t) = sum(strongClaResult(1:Npos)) / Npos;
% compute negative accuracy
negAccuracy(t) = sum(strongClaResult(Npos+1:end)) / Nneg;

%This is when the adaboost stops searching for features
if posAccuracy(t)==1 && negAccuracy(t) <= 0.5
break;
end

end

% Presenting update for the next cascaded iteration

% sort negative, if there is false deteciton, there will be 1 at the end
[sortedNeg, idxNeg] = sort(strongClaResult(Npos+1:end));
% get false detection negative index
for i = 1:Nneg
if sortedNeg(i) > 0
idxNeg = idxNeg(i:end);
break;
end
end
% get sample index for next cascaded iteration
idx = [1:Npos, Npos+idxNeg'];
% save trained data
%save(['strongCla_',num2str(stage),'.mat'],'strongCla','-mat', '-v7.3');
%save(['negAccuracy_',num2str(stage),'.mat'],'negAccuracy','-mat', '-v7.3');
% polarity, theta for each classifier
save(['ht_',num2str(stage),'.mat'],'ht','-mat', '-v7.3');
% alpha for each weak classifier
save(['alpha_',num2str(stage),'.mat'],'alpha','-mat', '-v7.3');
% indices for classifier h's feature
%save(['idxForNext',num2str(stage),'.mat'],'idx','-mat', '-v7.3');
% threshold for whole strong classifier --- may not be used
save(['threshold_',num2str(stage),'.mat'],'threshold','-mat', '-v7.3');
end



