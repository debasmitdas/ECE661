function [boxSum] = sumBox(I, box4)

%Given 4 corners in the integral image we have 
% to calculate the sum of pixels inside the box.

row_s=box4(1);
col_s=box4(2);
w=box4(3);
h=box4(4);

A = I(row_s, col_s);
B = I(row_s, col_s + w);
C = I(row_s+h, col_s);
D = I(row_s+h, col_s+w);

boxSum = A + D - (B+C);

end