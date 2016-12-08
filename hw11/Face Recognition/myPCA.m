function [normW, Neig]= myPCA(imgVec)
%Function to find PCA on image vectors

% Compute covariance matrix for sorting eigen values
[V,D]= eig(imgVec'*imgVec);
eigV = diag(D);
[~,idx] = sort(-1.0 .* eigV);
eigV = eigV(idx);
V = V(:,idx);

% For each image, we get the no. of eigenvectors which have 
% eigen values greater than 1
Neig=0;
for i=1:size(imgVec,2)
if eigV(i) > 1
Neig = Neig + 1;
end
end 

% We have to compute the weight matrix
w=imgVec*V;

%Next we normalize w
[r,c]=size(w);
normW=zeros(r,c);
for i=1:c
   normW(:,i)=w(:,i)/norm(w(:,i));
end
end

