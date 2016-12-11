% This is the main method for PCA

%Number of persons or the no. of classes
Npers=30;
%Number of trials per person
Ntrials=21;
%Path for training images
trPath = 'Face/train/';
tePath = 'Face/test/';
% load training images
[trainImg, ~, ~] = loadImg(trPath, Npers, Ntrials);  
% load testing images
[testImg, ~, ~] = loadImg(tePath, Npers, Ntrials);
% load trained w
[w, Neig] = myPCA(trainImg); % Taking the number of eigen values and 
% test using different number of eigenvectors, from small to large
accPCA = zeros(1, Neig);
TrainDataY=zeros(Npers*Ntrials,1);
for i=1:Npers*Ntrials
    TrainDataY(i,1)=ceil(i/Ntrials);
end


for i=1: Neig
partEig=w(:,1:i);

% project training images
 trainProj = zeros(i, Npers*Ntrials);
 for j = 1:Npers*Ntrials
 trainProj(:,j) = partEig' * trainImg(:,j);
 end

 % project testing images
 testProj = zeros(i, Npers*Ntrials);
 for j = 1:Npers*Ntrials
 testProj(:,j) = partEig' * testImg(:,j);
 end

TrainDataX=trainProj';
TestDataX=testProj';
%Training a K-Nearest neighbour K=1
mdl=fitcknn(TrainDataX, TrainDataY, 'NumNeighbors',1, 'distance', 'euclidean');
TestDataPred=mdl.predict(TestDataX);

%Testing using nearest neighbour and calculating accuracy
Diff=TestDataPred-TrainDataY;
accPCA(1,i)=nnz(~Diff);    

end 
% compute accuracy for different dim of eigen space
accPCA = accPCA / (Npers*Ntrials);
plot(accPCA);

