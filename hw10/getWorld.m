function [Xn] = getWorld (P1,P2,x1,x2)
% This code is for getting the World co-ordiantes
A = [ (x1(1)*P1(3,:) -P1(1,:));
(x1(2)*P1(3,:) -P1(2,:));
(x2(1)*P2(3,:) -P2(1,:));
(x2(2)*P2(3,:) - P2(2,:)) ];

[U D V]=svd(A);
Xn=V(:,4);
Xn=Xn/Xn(end);
Xn=Xn';

end