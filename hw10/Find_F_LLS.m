function [F]=Find_F_LLS(x1,x2,T1,T2)

n_corresp=size(x1,1);

%The correspondences are normalized;

x1t=(T1*x1')';
x2t=(T2*x2')';

A=zeros(n_corresp,9); 
 
for i=1:1:n_corresp
A(i,:) = [x2t(i,1)*x1t(i,1) x2t(i,1)*x1t(i,2) x2t(i,1) ...
x2t(i,2)*x1t(i,1) x2t(i,2)*x1t(i,2) ...
x2t(i,2) x1t(i,1) x1t(i,2) 1];
end

%Perform SVD on A
[U D V]=svd(A);

%F is the last column vector in V
F=reshape (V(:,end),3,3).';

%Condition the F Matrix
[UF DF VF] = svd(F);
DF(end,end)=0; % Make rank 2 by zeroing the last singular value

F=UF*DF*VF';
F=T2'*F*T1;
end







