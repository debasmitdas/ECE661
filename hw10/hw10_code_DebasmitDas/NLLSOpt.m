function [P2 F X]= NLLSOpt(errorFunchandle,x1,x2,P1,P2)
%First we need to create a parameter of vectors
% Total number of parameters should be 12+3*totalcoresspondences
size(P2);
p = [reshape(P2',1,12)];
n_corresp=size(x1,1);
X=zeros(n_corresp,4);
Xtmp=zeros(n_corresp,4);

for i=1:size(x1,1)
    Xn=getWorld(P1,P2,x1(i,:),x2(i,:));
    Xtmp(i,:)=Xn;
    p=[p Xn(1:3)];
end
p=double(p);
x1=double(x1);
x2=double(x2);

options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');

p_updated=lsqnonlin(errorFunchandle,p,[],[],options,x1,x2);

P2=reshape(p_updated(1:12),4,3)';
e2=P2(:,4);
ex=[0 -e2(3) e2(2); e2(3) 0 -e2(1); -e2(2) e2(1) 0];
M=P2(:,1:3);
F=ex*M;

% Return World 3D co-ordinates
cnt=13;
for i=1:n_corresp
    X(i,:)=[p_updated(cnt:cnt+2) 1];
    cnt=cnt+3;
end
end




