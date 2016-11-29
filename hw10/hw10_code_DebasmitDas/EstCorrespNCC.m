function [C] = EstCorrespNCC(feat1,feat2,valpts1,valpts2)

% The NCC Matrix is created
C=zeros(size(feat1,1),size(feat2,1));

for i=1:size(C,1)
    for j=1:size(C,2)
        C(i,j)=sum((feat1(i,:)-mean(feat1(i,:))).*(feat2(i,:)-mean(feat2(i,:)))) ...
            /sqrt(sum((feat1(i,:)-mean(feat1(i,:))).^2)*sum((feat2(i,:)-mean(feat2(i,:))).^2));
end
end
end