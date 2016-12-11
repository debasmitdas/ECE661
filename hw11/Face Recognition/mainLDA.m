% This is the main method for LDA
%Number of persons or the no. of classes
Npers=30;
%Number of trials per person
Ntrials=21;
%Path for training images
trPath = 'Face/train/';
tePath = 'Face/test/';
% load training images
[~,trainImg, meanTrain] = loadImg(trPath, Npers, Ntrials);
% load testing images
[~,testImg, meanTest] = loadImg(tePath, Npers, Ntrials);

% get trained data
[vecU, Z] = myLDA(trainImg,meanTrain,Npers,Ntrials);
% test using different number of eigenvalues
Neig = 30;
accLDA = zeros(1, Neig);

TrainDataY=zeros(Npers*Ntrials,1);
for i=1:Npers*Ntrials
    TrainDataY(i,1)=ceil(i/Ntrials);
end

for i = 1:Neig
 % compute part eigenvector U
 partVecU = vecU(:,1:i);
 W = Z * partVecU;

 % normalize W
 for j = 1:i
 W(:,j) = W(:,j) / norm(W(:,j));
 end

 % project training images
 trainProj = zeros(i, Npers*Ntrials);
 for j = 1:Npers*Ntrials
 trainProj(:,j) = W' * (trainImg(:,j)-meanTrain);
 end

 % project testing images
 testProj = zeros(i, Npers*Ntrials);
 for j = 1:Npers*Ntrials
 testProj(:,j) = W' * (testImg(:,j)-meanTest);
 end

TrainDataX=trainProj';
TestDataX=testProj';
%Training a Nearest neighbour
mdl=fitcknn(TrainDataX, TrainDataY, 'NumNeighbors',1, 'distance', 'euclidean');
TestDataPred=mdl.predict(TestDataX);

%Testing using nearest neighbour and calculating accuracy
Diff=TestDataPred-TrainDataY;
accLDA(1,i)=nnz(~Diff);    

end 
% compute accuracy for different dim of eigen space
accLDA = accLDA / (Npers*Ntrials);
plot(accLDA);
