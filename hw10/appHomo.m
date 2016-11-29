function [Xnew Hnew]=appHomo(H,X)
% This function is used to apply homography to an image 
% The output is the new image and the required homography
%X=single(X);
hOrig=size(X,1);
wOrig=size(X,2);

% To find the boundary of the resulting image
a=[1 1];
b=[wOrig 1];
c=[1 hOrig];
d=[wOrig hOrig];

i=H*[a';1];i=i/i(end); a_(1)=round(i(1)); a_(2)=round(i(2));
i=H*[b';1];i=i/i(end); b_(1)=round(i(1)); b_(2)=round(i(2));
i=H*[c';1];i=i/i(end); c_(1)=round(i(1)); c_(2)=round(i(2));
i=H*[d';1];i=i/i(end); d_(1)=round(i(1)); d_(2)=round(i(2));

tx1=min([a_(1) b_(1) c_(1) d_(1)]);
tx2=max([a_(1) b_(1) c_(1) d_(1)]);

ty1=min([a_(2) b_(2) c_(2) d_(2)]);
ty2=max([a_(2) b_(2) c_(2) d_(2)]);

% To find the height and width of projected image into the world plane
ht=(ty2-ty1);
wt=(tx2-tx1);

H_scale=[wOrig/wt 0 0;0 hOrig/ht 0;0 0 1];
H=H_scale*H;

i=H*[a';1];i=i/i(end); a_(1)=round(i(1)); a_(2)=round(i(2));
i=H*[b';1];i=i/i(end); b_(1)=round(i(1)); b_(2)=round(i(2));
i=H*[c';1];i=i/i(end); c_(1)=round(i(1)); c_(2)=round(i(2));
i=H*[d';1];i=i/i(end); d_(1)=round(i(1)); d_(2)=round(i(2));

tx1=min([a_(1) b_(1) c_(1) d_(1)]);
tx2=max([a_(1) b_(1) c_(1) d_(1)]);

ty1=min([a_(2) b_(2) c_(2) d_(2)]);
ty2=max([a_(2) b_(2) c_(2) d_(2)]);

%Now we have found the offsets
tx=tx1;
ty=ty1;

T=[1 0 -tx+1;0 1 -ty+1;0 0 1];
Hnew=T*H;
H_inv=Hnew^-1;

%Now the output is an image 
Xnew=zeros(hOrig,wOrig,3);
for m=1:hOrig
    for n=1:wOrig
        tmp=H_inv*[n;m;1];
        tmp=tmp/tmp(end);
        tmp=biLinear(tmp(1),tmp(2),X);
        
        Xnew(m,n,:)=tmp;
    end
end

Xnew=uint8(Xnew);
end

        





