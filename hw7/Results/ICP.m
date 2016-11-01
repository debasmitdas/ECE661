function Qtrans = ICP(Q, P)

%%%%%%%%%%%%% ICP algorithm %%%%%%%%%%%%%%
% 'moving' is the pointcloud to be transformed (Q)
% 'fixed' is the reference pointcloud (P)
% 'output' is transformed pointcloud (Q_transformed)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
th = 0.1^2;

Pprime = [];
Qprime = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1 : Find the closest points
for k = 1:size(Q,2)
    [D,I] = pdist2(P',Q(:,k)','euclidean','Smallest',1);
    if D < th
        Pprime = [Pprime,P(:,I)];
        Qprime = [Qprime,Q(:,k)];
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2 : Estimate R and t
N = size(Pprime,2);

Pc = sum(Pprime,2)/N;
Qc = sum(Qprime,2)/N;

Mp = Pprime-repmat(Pc,1,N);
Mq = Qprime-repmat(Qc,1,N);

C = Mq*Mp';

[U S V] = svd(C);

R = V*U';
t = Pc - R * Qc;

T = [R, t;0 0 0 1];

Qtrans = T*[Q;ones(1,size(Q,2))];
Qtrans = Qtrans./repmat(Qtrans(4,:),4,1);
Qtrans(4,:) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Pointcloud After ICP
figure;
scatter3(P(1,:),P(2,:),P(3,:),'.')
hold on
scatter3(Qtrans(1,:),Qtrans(2,:),Qtrans(3,:),'.')
hold off
xlim([-0.8 0.4])
ylim([-0.5 0.5])
zlim([0.4 1])
view(-80,8)
end

