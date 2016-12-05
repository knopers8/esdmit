function [ out ] = quad_kernel( X_i,b )
%quad kernel
 m=length(X_i);
 D=quad_kernel_size(m);
%  out =zeros(D,1) ;
out =zeros(m,1) ;
 for i=1:m
     out(i)=X_i(i)^2;
 end   
 for i=2:m % This double sum in middle
       for j=1:i-1
           out =[out ;sqrt(2)*X_i(i)*X_i(j)];        
       end       
 end  
 out2=zeros(m,1) ;
  for i=1:m
     out2(i)=sig_sqrt(2*b)*X_i(i);
  end   
%   out=[out; out2;b^2]';
 out=[out;1]';
end