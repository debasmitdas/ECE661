function [x1,x2]=GetFinFeatNCC(C,feat1,feat2,valpts1,valpts2,r_ncc,T_ncc,F,thresh_F)

x1=[];
x2=[];
lst=size(C,2);

for i=1:size(C,1)
    [b1,i1]=max(C(i,:));
    [b2,i2]=max(C(i,[(1:i1-1) i1+1:lst]));
    if (i2>=i1)
        i2=i2+1;
    end
    if (b1 <T_ncc)
        % Do not do anything;
    elseif ((b2/b1)>r_ncc)
        % Do nothing
    elseif (abs(norm(feat1(i,:)-feat2(i,:)))>50)
        % Do nothing
    else
        x1=[x1;valpts1.Location(i,1) valpts1.Location(i,2) 1];
        x2=[x2;valpts2.Location(i,1) valpts2.Location(i,1) 1];
    end
end
end

% This is for declaring the epipole constraints
function [c]=epi_constraint(a,b,c,d,F,thresh)
x1=double([a,b,1]');
x2=double([c,d,1]');
result=x1'*F*x1;
if (result < thresh)
    c=1;
else
    c=0;
end
c=1;
end