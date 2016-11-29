function []=PlotWorPts(X,x1f,x2f,P1,P2,F,thresh_F)
x_w=[];
%Triangulation is done here
for i=1:size(X,1)
    x1=(P1*X(i,:)')';
    x1=x1/x1(end);
    x2=(P2*X(i,:)')';
    x2=x2/x2(end);
    x_w=[x_w;X(i,:)];
end
figure;
scatter3(x_w(:,1),x_w(:,2),x_w(:,3),'o', 'Linewidth',2);
disp(['Total points in world 3D = ' num2str(size(x_w,1))]);
end
    