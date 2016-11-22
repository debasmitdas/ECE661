function [corner] = findCorners(filename)
gr_truth=imread(filename);
gr_truth_gray = rgb2gray(gr_truth);
gr_truth_edge = edge(gr_truth_gray,'canny',0.7);%for provided dataset 0.8
figure
imshow(gr_truth_edge)

[H, T, R] = hough(gr_truth_edge,'RhoResolution',0.5); %for provided dataset 0.5

P = houghpeaks(H,18,'Threshold',15); %for our and provided dataset 18 and 15
%There will be 18 lines
lines = houghlines(gr_truth_edge,T,R,P,'FillGap',150,'MinLength',70);
%for our and provided dataset 150 and 70
line_param = zeros(length(lines),2); %slope, y-intersect,
%Initializing the horizontal and vertical
hor = []; ver = [];


for k = 1:length(lines)
 xypt = [lines(k).point1; lines(k).point2];
 %find the equation of the line y = mx + b
 line_param(k,1) = (xypt(1,2)-xypt(2,2))/(xypt(1,1)-xypt(2,1));
 % plot_line(lines,k,size(gr_truth_edge));
 if(abs(line_param(k,1))>1)
 ver = [ver k];
 else
 hor = [hor k];
 end
 if(abs(line_param(k,1)) == inf)
 line_param(k,2) = inf;
 else
 line_param(k,2) = xypt(1,2) - line_param(k,1)*xypt(1,1);
 end
end
%Initializing the list for the corners
corner = [];
for i = 1:length(lines)
 n_c{i} = [];
end

%This is used to get rid of the extra lines
lines_hor = lines(hor);
ehor = zeros(1,length(hor));
for i= 1:length(lines_hor)
 for j = i+1:length(lines_hor)
 [pt]= intsect(lines_hor(i), lines_hor(j));
 if(pt(1)>1 && pt(1)<size(gr_truth,2) && pt(2)>1 && pt(2)<size(gr_truth,1))
 ehor(i) =ehor(i)+ 1;
 ehor(j) = ehor(j)+1;
 end
 end
end
lines_ver = lines(ver);
ever = zeros(1,length(ver));
for i= 1:length(lines_ver)
 for j = i+1:length(lines_ver)
 [pt]= intsect(lines_ver(i), lines_ver(j));
 if(pt(1)>1 && pt(1)<size(gr_truth,2) && pt(2)>1 && pt(2)<size(gr_truth,1))
 ever(i) = ever(i) +1;
 ever(j) = ever(j) +1;
 end
 end
end
%Sorting the line according to indices 
[ever ind1] = sort(ever,'ascend');
[ever ind2] = sort(ehor,'ascend');
lines = lines([hor(ind2(1:10)) ver(ind1(1:8))]);

%plot the lines
figure
imshow(gr_truth_gray)
for k = 1:length(lines)
 ptxy = [lines(k).point1; lines(k).point2];
 %find the equation of the line y = mx + b
 %find slope m
 line_param(k,1) = (ptxy(1,2)-ptxy(2,2))/(ptxy(1,1)-ptxy(2,1));
 if(abs(line_param(k,1)) == inf)
 line_param(k,2) = inf;
 hold on
 y = 1:size(gr_truth,1);
 x = ptxy(1,1)*ones(1,length(y));
 plot(x,y,'Color','green')
 else
 line_param(k,2) = ptxy(1,2) - line_param(k,1)*ptxy(1,1);
 f = @(x) line_param(k,1)*x + line_param(k,2);
 x = 1:size(gr_truth,2);
 y = uint64(f(x));
 hold on
 plot(x,y,'Color','green');
 end
end

%*********************************************************************
%find the corners
for i= 1:length(lines)
 for j = i+1:length(lines)
 [pt]= intsect(lines(i), lines(j));
 if(pt(1)>1 && pt(1)<size(gr_truth,2) && pt(2)>1 && pt(2)<size(gr_truth,1))
 corner = [corner; pt ];
% hold on
% plot(pt(1),pt(2),'r*')
 n_c{i} = [n_c{i} size(corner,1)];
 n_c{j} = [n_c{j} size(corner,1)];
 end
 end
end

%label the corners same way 
hor = [];
ver = [];
for i = 1:length(lines)
 if(length(n_c{i}) ==8)
 hor = [hor i];
 else
 ver = [ver i];
 end
end
xs = zeros(length(ver),1); 
for i = 1:length(ver)
 %sort corners according to smallest x
 ind = n_c{ver(i)}; %these are corners that are on that line
 xs(i) = min(corner(ind,1)); %this is the smallest y for that vertical line
end
[d ind] = sort(xs,'ascend'); %sort vertical lines according to the smalles x
ver = ver(ind); %vertical lines are sorted
labels = zeros(80,1);
cnt = 0;
ys = zeros(10,1); %for each vertical line sort
for i = 1:length(ver)
 ind = n_c{ver(i)}; %these are corners that are on that line
 ys = corner(ind,2);
 [d sind] = sort(ys,'ascend');
 for j = 1:length(sind) %1 to 10
 cnt =cnt + 1;
 labels(cnt) = ind(sind(j));
 end
end
corner = corner(labels,:);

% This script is for labelling the corners
for i = 1:length(labels)
 hold on
 text(corner(i,1),corner(i,2),int2str(i),'Color','r');
end
end