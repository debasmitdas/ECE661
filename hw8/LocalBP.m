function f=LocalBP(img,R,P)
% The output f is the LocalBP feature representation
% The input img is the image
% R is the radius of the circular pattern
% P is the number of points to sample on the circle
imgg=rgb2gray(img);
f=zeros(P+2,1);

for i=R+1:size(img,1)-(R+1)
    for j=R+1:size(img,2)-(R+1)
        patt=[];
        %disp(i);
        %disp(j);
        
        for p=0:P-1
            delk=R*cos(2*pi*p/P);
            dell=R*sin(2*pi*p/P);
            if abs(delk) < 0.001
                delk=0;
            end
            if abs(dell) < 0.001
                dell=0;
            end
            % To do Bilinear transformation
            k=i+delk; l=j+dell;
            kbase=round(k);  lbase=round(l);
            deltak=k-kbase; deltal=l-lbase;
            if (deltak<0.001) && (deltal<0.001) 
                imgp=img(kbase,lbase);
            elseif (deltal<0.001)
                imgp=(1-deltak)*imgg(kbase,lbase) + deltak*imgg(kbase+1,lbase);
            elseif (deltak<0.001)
                imgp=(1-deltal)*imgg(kbase,lbase) + deltal*imgg(kbase,lbase+1);
            else
                imgp=(1-deltak)*(1-deltal)*imgg(kbase,lbase) + ...
                     (1-deltak)*deltal*imgg(kbase,lbase+1) + ...
                     deltak*deltal*imgg(kbase+1,lbase+1 ) + ...
                     deltak*(1-deltal)*imgg(kbase+1,lbase);
            end
            
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
        
        if (sum(minpatt)==0)
            f(1)=f(1)+1;
        elseif (sum(minpatt)==P)
            f(P+1)=f(P+1)+1;
        elseif (number_runs_zeros > 2) || (number_runs_ones >2)
            f(P+2)=f(P+2)+1;
        else
            f(sum(minpatt))=f(sum(minpatt))+1;
        end
    end
end

f=f/sum(f);

        
        
        
        
        
            
            
                
            
            
            
