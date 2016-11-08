
nneighbors=5; % no. of neighbors for k-nearest neighbors
R=1; % radius
P=8; % no. of points on the circle
%Processing the training data
TrainDataX=[];
TrainDataY=[];
%Looping over all the building images 
files=dir('imagesDatabaseHW8/training/building/*.jpg');
img=cell(20,1);
i=1;
for file=files'
    img{i}=imread(strcat('imagesDatabaseHW8/training/building/',file.name));
    i=i+1;
end
parfor i=1:20
    f=LocalBP(img{i},R,P);
    TrainDataX=[TrainDataX;f'];
    TrainDataY=[TrainDataY;1];
end


%Looping over all the car images 
files=dir('imagesDatabaseHW8/training/car/*.jpg');
img=cell(20,1);
i=1;
for file=files'
    img{i}=imread(strcat('imagesDatabaseHW8/training/car/',file.name));
    i=i+1;
end
parfor i=1:20
    f=LocalBP(img{i},R,P);
    TrainDataX=[TrainDataX;f'];
    TrainDataY=[TrainDataY;2];
end


%Looping over all the mountain images 
files=dir('imagesDatabaseHW8/training/mountain/*.jpg');
img=cell(20,1);
i=1;
for file=files'
    img{i}=imread(strcat('imagesDatabaseHW8/training/mountain/',file.name));
    i=i+1;
end
parfor i=1:20
    f=LocalBP(img{i},R,P);
    TrainDataX=[TrainDataX;f'];
    TrainDataY=[TrainDataY;3];
end

%Looping over all the tree images 
files=dir('imagesDatabaseHW8/training/tree/*.jpg');
img=cell(20,1);
i=1;
for file=files'
    img{i}=imread(strcat('imagesDatabaseHW8/training/tree/',file.name));
    i=i+1;
end
parfor i=1:20
    f=LocalBP(img{i},R,P);
    TrainDataX=[TrainDataX;f'];
    TrainDataY=[TrainDataY;4];
end


%Processing the testing data
TestDataX=[];
TestDataY=[];
%Looping over all the building images 
files=dir('imagesDatabaseHW8/testing/building*.jpg');
img=cell(5,1);
i=1;
for file=files'
    img{i}=imread(file.name);
    i=i+1;
end
parfor i=1:5
    f=LocalBP(img{i},R,P);
    TestDataX=[TestDataX;f'];
    TestDataY=[TestDataY;1];
end


%Looping over all the car images 
files=dir('imagesDatabaseHW8/testing/car*.jpg');
img=cell(5,1);
i=1;
for file=files'
    img{i}=imread(file.name);
    i=i+1;
end
parfor i=1:5
    f=LocalBP(img{i},R,P);
    TestDataX=[TestDataX;f'];
    TestDataY=[TestDataY;2];
end

%Looping over all the mountain images 
files=dir('imagesDatabaseHW8/testing/mountain*.jpg');
img=cell(5,1);
i=1;
for file=files'
    img{i}=imread(file.name);
    i=i+1;
end
parfor i=1:5
    f=LocalBP(img{i},R,P);
    TestDataX=[TestDataX;f'];
    TestDataY=[TestDataY;3];
end


%Looping over all the tree images 
files=dir('imagesDatabaseHW8/testing/tree*.jpg');
img=cell(5,1);
i=1;
for file=files'
    img{i}=imread(file.name);
    i=i+1;
end
parfor i=1:5
    f=LocalBP(img{i},R,P);
    TestDataX=[TestDataX;f'];
    TestDataY=[TestDataY;4];
end


% Snippet for plotting the histogram

% For building
figure;
bar(TrainDataX(1,:))
title('Building LBP Histogram','FontWeight','bold')

% For car
figure;
bar(TrainDataX(21,:))
title('Car LBP Histogram','FontWeight','bold')

% For mountain
figure;
bar(TrainDataX(41,:))
title('Mountain LBP Histogram','FontWeight','bold')

% For tree
figure;
bar(TrainDataX(61,:))
title('Tree LBP Histogram','FontWeight','bold')

% Snippet for plotting histogram ends

%Training a K-Nearest neighbour
mdl=fitcknn(TrainDataX, TrainDataY, 'NumNeighbors',nneighbors, 'distance', 'euclidean', 'Standardize',1);

%Testing a K-Nearest Neighbor
TestDataPred=mdl.predict(TestDataX);

%Confusion Matrix is plotted
Target=zeros(4,size(TestDataX,2));
Pred=zeros(4,size(TestDataX,2));

for i=1:size(TestDataX)
    Target(TestDataY(i),i)=1;
    Pred(TestDataPred(i),i)=1;
end
figure;
plotconfusion(Pred,Target);
xlabel('Output Class','FontWeight','bold')
ylabel('Target Class','FontWeight','bold')




