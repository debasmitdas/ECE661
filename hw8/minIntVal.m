function pout=minIntVal(pin,P)
A=size(P,1);
minshift=P+1;
for i=1:P
    %Looking over all circular shifts
    Y=circshift(pin,i);
    
    A(i)=bi2de(Y','left-msb');
end
[~,minshift]=min(A); % Finding the index of the one with minIntVal

pout=circshift(pin,minshift);
end    
    

