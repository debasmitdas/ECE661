function [T] = norm_point(x)

% The 8-point normalization algorithm is applied
mean_x=mean(x(:,1));
mean_y=mean(x(:,2));

std_tmp = 0;
for i = 1:size(x,1)
std_tmp = std_tmp + sqrt((x(i,1)-mean_x)^2+(x(i,2)-mean_y)^2);
end

std_tmp=std_tmp/size(x,1);
scale=sqrt(2)/std_tmp;
xtr= -scale*mean_x;
ytr= -scale*mean_y;
T=[scale 0 xtr; 0 scale ytr; 0 0 1];
end


