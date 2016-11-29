function []=plot_corresp(im1,im2,x1,x2)

im=[im1 im2];
imshow(im);
hold on
n_rows=size(im1,1);
n_cols=size(im1,2);

for i=1:size(x1,1)
    plot([x1(i,1)/x1(i,3);x2(i,1)/x2(i,3)+n_cols], [x1(i,2)/x1(i,3);x2(i,2)/x2(i,3)], 'Color','r','linewidth',3)
    scatter([x1(i,1)/x1(i,3);x2(i,1)/x2(i,3)+n_cols], [x1(i,2)/x1(i,3);x2(i,2)/x2(i,3)], 'blue', 'filled', 'linewidth', 5)
end

hold off

