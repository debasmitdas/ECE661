
load fisheriris
X = meas;
Y = zeros(150,1);
Y(1:50,:)=1;
Y(51:100,:)=2;
Y(101:150,:)=3;
Mdl = fitcknn(X,Y,'NumNeighbors',4);
Out=Mdl.predict(X);

% 
% for i=1:80
% if (TrainDataY(i)==1)
% TrainDataY(i)==0001
% elseif (TrainDataY(i)==2)
% TrainDataY(i)==0010
% elseif (TrainDataY(i)==3)
% TrainDataY(i)=0100
% else
% TrainDataY(i)=1000
% end
% end



