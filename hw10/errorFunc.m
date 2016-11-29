function e=errorFunc(p,x1,x2)

P2=reshape(p(1:12),4,3)';
n_corresp=size(x1,1);
% Return World 3D co-ordinates
cnt=13;
for i=1:n_corresp
    X(i,:)=[p(cnt:cnt+2) 1];
    cnt=cnt+3;
end
e=0;
% Error func is described that is to be used for LM refinement
for i=1:n_corresp
    x1est=([eye(3,3) zeros(3,1)]*X(i,:)')'
    x1est=x1est/x1est(end);
    x2est=(P2*X(i,:)')'
    x2est=x2est/x2est(end);
    e=e+(norm(x1(i,:)-x1est))^2 + (norm(x2(i,:)-x2est))^2
end
