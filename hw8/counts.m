function [a,b]=counts(minpatt)        

v=minpatt';
        % Code to find the number of runs of zeros and ones
        w=[1 v 1];
        runs_zeros = find(diff(w)==1)-find(diff(w)==-1);
        number_runs_zeros = length(runs_zeros);
        number_runs_ones = number_runs_zeros-1+v(1)+v(end);
        
        a=number_runs_zeros;
        b=number_runs_ones;