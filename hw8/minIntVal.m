function pout=minIntVal(pin,P)
pin=pin';
min=2^P-1;
minshift=P+1;
for i=0:P-1
    
    Y=circshift(pin',i);
    if(bi2de(Y')<min)
        min=bi2de(Y');
        minshift=i;
    end
end

pout=flipud(circshift(pin',minshift));


    
    

