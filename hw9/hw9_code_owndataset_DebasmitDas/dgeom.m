function err = dgeom(p,xW,xIM,rad_dist,nimg)
% p is the set of parameters for LM algorithm
ax = p(1);
s = p(2);
x0 = p(3);
ay = p(4);
y0 = p(5);
K = [ax s x0; 0 ay y0; 0 0 1]; % The intrinsic calibration matrix
if(rad_dist == 1)
 k1 = p(6);
 k2 = p(7); % These are the parameters of radial distortion
 K1 = [ax 0 x0; 0 ay y0; 0 0 1];
 cnt = 7;
else
 cnt = 5;
end
xproj = zeros(1,nimg*160);
n1=1;
for k = 1:nimg
 % Converting to the R,t using Rodriguez formula
 w = p(cnt+1:cnt+3);
 t = p(cnt+4:cnt+6)';
 cnt = cnt + 6;
 wx = [0 -w(3) w(2); w(3) 0 -w(1); -w(2) w(1) 0];
 phi = norm(w);
 R = eye(3)+sin(phi)/phi*wx + (1-cos(phi))/phi*wx^2;
 n2=1;
 for i = 1:80
 % Projection for all the corner points onto the fixed image.  
 x = K*[R t]*[xW(n2:n2+1) 0 1]';
 xproj(n1:n1+1) = [x(1)/x(3) x(2)/x(3)];
 if(rad_dist == 1)
xp = [xproj(n1:n1+1) 1];
xw = inv(K1)*xp';
r2 = xw(1)^2 + xw(2)^2;
xp1 = xw(1) + xw(1)*(k1*r2+k2*r2^2);
xp2 = xw(2) + xw(2)*(k1*r2+k2*r2^2);
x = K1*[xp1 xp2 1]';
xproj(n1:n1+1) = [x(1)/x(3) x(2)/x(3)];
 end
 n1 = n1+2;
 n2 = n2+2;
 end
end
err = xIM - xproj;
end
