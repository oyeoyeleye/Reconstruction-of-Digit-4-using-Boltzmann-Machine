[a,b]=size(orig);
N=b;
epsilon=0.01;
epoch=5;
update_size=7;
H=28*2;
out=28*8;
W=initweights(280*2,N,H);


for p=1:a
%%%%%%%%binarize input
for i=1:b
   if orig(p,i)<=0;
   img(i)=0;
   else
   img(i)=1;
   end
end

%%%%initialize configuration of neurons and weights%%%%%%
x=zeros(1,N+H);
counter=0;
for i=1:N
taboo=[281:504];
    if not(ismember(i,taboo))
    counter=counter+1;
    x(counter)=img(i);    
    end 
end
f_current=x;
c_current=x;
for i=1:28*8
c_current(i+counter+H)=img(280+i);    
end    

f_record=zeros(update_size,N+H); 
c_record=zeros(update_size,N+H);

for l=1:epoch
count=0;
order=counter+randperm(H);
f_current=generate(f_current,counter+H+1,W);    
r_image=[x(1:280), f_current(counter+H+1:N+H), x(281:counter)];       
imshow(reshape(r_image,28,28)')
    for i=1:length(order)
    %%%%%%simulate for free case%%%%%%%%% 
    f_ener=energy(W,order(i),f_current);
    f_alpha=1/(1+exp(-f_ener));
       if rand<f_alpha
       f_current(order(i))=1;
       else
       f_current(order(i))=0;
       end
    
    %%%%%%simulate for clamped case%%%%%%%%%
    c_ener=energy(W,order(i),c_current);
    c_alpha=1/(1+exp(-c_ener));
       if rand<c_alpha
       c_current(order(i))=1;
       else
       c_current(order(i))=0;
       end
      
       if mod(i,update_size)==0
       f_current=generate(f_current,counter+H+1,W);    
       count=count+1;    
       j=update_size;
       disp(sprintf('Image %d: Epoch %d: updating weight %d th time',p,l,count)) 
       else
       j=mod(i+update_size,update_size);
       end 
       f_record(j,:)=f_current; 
       c_record(j,:)=c_current;
   
   %%%%%weights update%%%%%%%%%%
       if mod(i,update_size)==0
       free=zeros(N+H,N+H,update_size);
       clamped=zeros(N+H,N+H,update_size);
           for ii=1:N+H 
                for jj=counter+1:N+H
                    for kk=1:update_size
                        if not(ii==jj) & ii<=counter+H
                        free(ii,jj,kk)=coact(f_record(kk,ii),f_record(kk,jj));      
                        clamped(ii,jj,kk)=coact(c_record(kk,ii),c_record(kk,jj));
                        end  
                    end     
                    if not(ii==jj) & ii<=counter+H
                    W(ii,jj)=W(ii,jj)+epsilon*(mean(clamped(ii,jj,:))-mean(free(ii,jj,:)));     
                    W(jj,ii)=W(ii,jj);
                    end
                end
           end  
       end  
          
    end
end    
end

f_current=generate(f_current,counter+H+1,W);    
r_image=[x(1:280), f_current(counter+H+1:N+H), x(281:counter)];       
imshow(reshape(r_image,28,28)')

function e=energy(W,i,x)
e=0;
[a,b]=size(W);
for j=1:b
   e=e+W(i,j)*x(j);    
end
end   
    

function W=initweights(counter,N,H)
W=zeros(N+H,N+H);
    for i=1:counter+H
        for j=counter+1:N+H
            if not(i==j) &  i<=counter+H
            W(i,j)=random('Normal',0,0.1)/sqrt(784);
            end
            
        end
    end    
W=W+W';
end


function a=generate(x,c,W)
a=x;
for i=c:length(x) 
ener=energy(W,i,x);
alpha=1/(1+exp(-ener));
       if rand<alpha
       a(i)=1;
       else
       a(i)=0;    
       end    
end
end

function a=coact(x,y)
    a=y*x;
end

