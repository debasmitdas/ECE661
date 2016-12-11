function str = num2str2digit(num)
% This function is used for converting
% indices in images to strings 
if num<10
str = ['0',num2str(num)];
else
str = num2str(num);
end
end
