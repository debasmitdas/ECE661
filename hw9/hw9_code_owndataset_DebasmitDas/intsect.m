function [point]=intsect(l1, l2)
% This function is used to find the intersection of 2 lines
pt1 = [l1.point1 1];
pt2 = [l1.point2 1];
lA = cross(pt1,pt2); % Getting the first line
pt1 = [l2.point1 1];
pt2 = [l2.point2 1];
lB = cross(pt1,pt2); % getting the second line
point = cross(lA,lB); % getting the intersectiion of the two lines
point = double([point(1)/point(3) point(2)/point(3)]); % Converting to 
% Homogeneous coordinates
end