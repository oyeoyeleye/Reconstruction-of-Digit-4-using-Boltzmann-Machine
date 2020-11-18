%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save complete testing image in a variable "test". No need to remove
% middle four lines, the program does it for you.
% 
% Write the following in the command line to perform the desired task after
% running this code.
%
%1. display complete source image
%    imshow(reshape(img,28,28)')
%
%2. display incomplete test image:
%    imshow(reshape(preximg,28,28)')
%
%3. display reconstructed image:
%    imshow(reshape(r_image,28,28)')
test=orig(1,:);
%%%%%%%%binarize input
for i=1:length(test)
   if test(i)<=0;
   img(i)=0;
   else
   img(i)=1;
   end
end

%%%%construct incomplete image%%%%%%
prex=zeros(1,N+H);
counter=0;
for i=1:N
taboo=[281:504];
    if not(ismember(i,taboo))
    counter=counter+1;
    prex(counter)=img(i);   
    end 
end
preximg=[prex(1:280), prex(counter+H+1:N+H), prex(281:counter)];  
f_current=prex;
%%%%%%simulate for free case%%%%%%%%%
for i=1:H
ener=energy(W,counter+i,f_current);
f_alpha=1/(1+exp(-f_ener));
       if rand<f_alpha
       f_current(counter+i)=1;
       else
       f_current(counter+i)=0;
       end
end
f_current=generate(f_current,counter+H+1,W);    
r_image=[x(1:280), f_current(counter+H+1:N+H), x(281:counter)];       

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

function e=energy(W,i,x)
e=0;
[a,b]=size(W);
for j=1:b
   e=e+W(i,j)*x(j);    
end
end   