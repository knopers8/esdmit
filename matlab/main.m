close all;
choose_kernel=1; % 1 - linear, 0 - quadratic

if choose_kernel
    % Training data
     y =[   1     1     2     2     2   2 2 ]';
     X =[1     1
         2     2
         5     5
         6     6
         7     7
         16    15
         15    16];
     % Test data
     y_v=[1 1 2 2 2  ]';
     X_v=[1 2
           2 1
           5 6
           6 5
           15 15];
     C = 10 ;    
     [Xn,means, stds] = normalization( X );
     [N,m]=size(Xn);
     w0=rand(m+1,1); %Starting point
     fun_handle=@lin_kernel;
else     
       % Training data
     y =[ 1  1  1   2   2  2  2  ]';
     X =[0     0    
         1     2
         2     -1
         6     -5
         -6     6
         7     -7
         9    8];
     % Test data
     y_v=[1  2 ]';
     X_v=[1 0
           7 -8];
    C = 10 ; 
  [Xn,means, stds] = normalization( X );
   [N,m]=size(Xn);
    D=quad_kernel_size(m);
    w0=rand(D+1,1); %Starting point
    fun_handle=@quad_kernel;
end

max_it=10000; % max iterations
eps=0.001; % stop algorithm when eraror is below this value
[ classificator, output,error] = train_multi_svm( Xn,y, w0 ,C,max_it,eps,fun_handle);
[ X_vn] = classificator_normalization( X_v , means, stds );

[output, value] = classify_multi_svm( X_vn, classificator,fun_handle);
output