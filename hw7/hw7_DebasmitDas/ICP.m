function Qt = ICP(Q, P, th)

%Output is the transformed target point cloud for one iteration

Pp = [];
Qp = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find the closest points
for k = 1:size(Q,2)
    [D,I] = pdist2(P',Q(:,k)','euclidean','Smallest',1);
    if D < th
        Pp = [Pp,P(:,I)];
        Qp = [Qp,Q(:,k)];
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Estimate Transformation Matrix
N = size(Pp,2);

Pc = sum(Pp,2)/N;
Qc = sum(Qp,2)/N;

Mp = Pp-repmat(Pc,1,N);
Mq = Qp-repmat(Qc,1,N);

C = Mq*Mp';

[U S V] = svd(C);

R = V*U';
t = Pc - R * Qc;

T = [R t;0 0 0 1];

Qt = T*[Q;ones(1,size(Q,2))];
Qt = Qt./repmat(Qt(4,:),4,1);
Qt(4,:) = [];

end
