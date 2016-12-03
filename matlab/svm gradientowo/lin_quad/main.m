close all;
kernel=1; % 1 - linear, 0 - quadratic

if kernel
     y =[   1     1     2     2     2   3]';
     X =[1     1
         2     2
         5     5
         6     6
         7     7
         15    15];
     C = 10 ;    
     [Xn] = normalization( X );
     [N,m]=size(Xn);
     w0=rand(m+1,1); %Starting point
     fun_handle=@lin_kernel;
else     
   error('Quadratic data no ready') 
   [Xn] = normalization( X );
   [N,m]=size(Xn);
    D=quad_kernel_size(m);
    w0=rand(D+1,1); %Starting point
    fun_handle=@quad_kernel;
end

max_it=10000; % max iterations
eps=0.001; % stop algorithm when error is below this value
[ output,error] = multi_svm( Xn,y, w0 ,C,max_it,eps,fun_handle)