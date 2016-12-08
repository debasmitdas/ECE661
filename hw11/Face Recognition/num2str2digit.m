function str = num2str2digit(num)
% This function is used for converting
% to numbers from string fro the purpose of
% loading images 
if num<10
str = ['0',num2str(num)];
else
str = num2str(num);
end
end