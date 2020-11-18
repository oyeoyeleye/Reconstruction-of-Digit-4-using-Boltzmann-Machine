%input is a random graph with adjacency A and weights matrix W
N=length(A);
epoch=100;
M=10 %keep track every M visits
order=randperm(N);
x=initial(N);
%W=initweights(A);
energy(x,W,1);

count=0;    %counter for energy recorded
count1=0;   %counter for global step
ener=zeros(epoch*N/M,1);
for k = 1:epoch
T=0.95^k;    
    for i=1:N
    count1=count1+1;    
    y=x;    
    j=order(i);
        y(j)=1-x(j);
        a=energy(x,W,1);
        b=energy(y,W,1);
        delta=b-a;
        
        if or(delta<0,rand<exp(-delta/T))
            x(j)=y(j);    
        %    display('accept')
        %else
        %    display('reject')
        end
        
        if mod(count1,10)==0
        count=count+1;   
        ener(count)=energy(x,W,1);
        end
        
    end   %end of i
end   %end of k    
display('total energy')
energy(x,W,1)
plot(ener)
%histogram(ener)
mean(ener)

function W=initweights(A)
N=length(A);
W=zeros(N,N);
    for i=1:N
        for j=1:N
            if A(i,j)==1 & j>i
            W(i,j)=-1 + (1+1)*rand(1,1);    
            end
        end
    end    
W=W+W';
end

function a=initial(N)
    for i=1:N
    a(i)=random('Binomial',1,0.5);
    end
end

function a=energy(X,W,T)
N=length(X);
a=0;
for i=1:N
    for j=1:N
    a=a+W(i,j)*X(i)*X(j)/T;
    end
end
end

