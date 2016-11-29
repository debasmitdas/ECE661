function [f]=biLinear(x,y,Z)
% Z is the image from where we got values 
% x,y is the location form which we need to do bilinear interpolation
S = size(Z);
if ((y<1)||(y>S(1)) )
f=zeros(1,1,3);
elseif ((x<1)||(x>S(2)))
f = zeros(1,1,3);
else
x1 = floor(x);
x2 = ceil(x);
y1 = floor(y);
y2 = ceil(y);
f = ( (x2-x)*(y2-y) ) / ( (x2-x1)*(y2-y1) ) * Z(y1,x1,:) + ...
( (x-x1)*(y2-y) ) / ( (x2-x1)*(y2-y1) ) * Z(y1,x2,:) + ...
( (x2-x)*(y-y1) ) / ( (x2-x1)*(y2-y1) ) * Z(y2,x1,:) + ...
( (x-x1)*(y-y1) ) / ( (x2-x1)*(y2-y1) ) * Z(y2,x2,:);
end
end