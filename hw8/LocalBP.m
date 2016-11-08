function f=LocalBP(img,R,P)
% The output f is the LocalBP feature representation
% The input img is the image
% R is the radius of the circular pattern
% P is the number of points to sample on the circle
imgg=rgb2gray(img);
f=zeros(P+2,1);

for i=R+1:size(imgg,1)-R
    for j=R+1:size(imgg,2)-R
        patt=[];
        
        
        for p=1:P
            % Neighbours are calculated
            delk=R*cosd(360*p/P);
            dell=R*sind(360*p/P);
            % To do Bilinear transformation
            k=i+delk; l=j+dell;
            kbase=floor(k);  lbase=floor(l);
            deltak=k-kbase; deltal=l-lbase;
            if (deltak==0 && deltal==0) 
                imgp=img(kbase,lbase);
            elseif (deltal==0)
                imgp=(1-deltak)*imgg(kbase,lbase) + deltak*imgg(kbase+1,lbase);
            elseif (deltak==0)
                imgp=(1-deltal)*imgg(kbase,lbase) + deltal*imgg(kbase,lbase+1);
            else
                imgp=(1-deltak)*(1-deltal)*imgg(kbase,lbase) + ...
                     (1-deltak)*deltal*imgg(kbase,lbase+1) + ...
                     deltak*deltal*imgg(kbase+1,lbase+1 ) + ...
                     deltak*(1-deltal)*imgg(kbase+1,lbase);
            end
            
            % Pattern is updated
            if (imgp >=imgg(i,j))
                patt=[patt;1];
            else
                patt=[patt;0];
            end
        end
        
        minpatt=minIntVal(patt,P);
        v=minpatt';
        % Code to find the number of runs of zeros and ones
        w=[1 v 1];
        runs_zeros = find(diff(w)==1)-find(diff(w)==-1);
        number_runs_zeros = length(runs_zeros);
        number_runs_ones = number_runs_zeros-1+v(1)+v(end);
        
        % Proper Encoding is done
        if ((number_runs_zeros + number_runs_ones)> 2) 
            f(P+2)=f(P+2)+1;
        else
            f(sum(minpatt)+1)=f(sum(minpatt)+1)+1;
        end
    end
end

f=f/sum(f);
end

        
        
        
        
        
            
            
                
            
            
            
