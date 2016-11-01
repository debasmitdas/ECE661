%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read Depth Data
Dp = dlmread('depthImage1ForHW.txt');
Dq = dlmread('depthImage2ForHW.txt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting Depth Image
figure;
imagesc(Dp)
figure;
imagesc(Dq)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Convert Depth Image to Point Cloud
% This is the camera calibration matrix
K = [365 0 256; 0 365 212; 0 0 1];
%This is the threshold for the ICP algorithm
th = 0.1;
%This is the number of iterations 
M=20;

w = size(Dp,2);
l = size(Dp,1);

P = [];
Q = [];

for i = 1:w
    for j = 1:l
        if Dp(j,i) ~= 0
            P = [P,Dp(j,i)*inv(K)*[j;i;1]];
        end
        
        if Dq(j,i) ~= 0
            Q = [Q,Dq(j,i)*inv(K)*[j;i;1]];
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Pointcloud Before ICP
figure;
scatter3(P(1,:),P(2,:),P(3,:),'.')
hold on
scatter3(Q(1,:),Q(2,:),Q(3,:),'.')
hold off
xlim([-0.9 0.5])
ylim([-0.5 0.5])
zlim([0.5 1])
view(-80,9)

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Conduct ICP
% 

for i = 1:M
    %Do the iterative ICP
    Q = ICP(Q,P,th);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Pointcloud after ICP
figure;
scatter3(P(1,:),P(2,:),P(3,:),'.')
hold on
scatter3(Q(1,:),Q(2,:),Q(3,:),'.')
hold off
xlim([-0.9 0.5])
ylim([-0.5 0.5])
zlim([0.5 1])
view(-80,9)







