function [A] = findA(xW,yW,xIM,yIM)
A = [];
for i = 1:length(xW)
 B = [xW(i) yW(i) 1 0 0 0 -xW(i)*xIM(i) -yW(i)*xIM(i) -xIM(i);
 0 0 0 xW(i) yW(i) 1 -xW(i)*yIM(i) -yW(i)*yIM(i) -yIM(i)];
 A = [A; B];
end
end