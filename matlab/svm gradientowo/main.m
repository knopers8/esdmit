%http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf pages 
% Fin minimum of cost function with gradient algorithm
% min w^t*w w=[w_1 w_1 .... w_m b]
% f(x)=w(1:m)'*x+b
close all;

X=[1 3
   2 2
   3 1
   3 2
   6 1
   5 2
   4 3
   3 4
   1 1];
y=[1; 1; 1; 1; -1; -1; -1; -1; -1];

[N,m]=size(X);
w0=rand(m+1,1); %Starting point


% X=[1 2
%    2 1
%    3 4
%    4 3 ];
% X=[1 
%    2 
%    3 
%    4  ];
  
% y=[-1; -1; 1; 1];

C=10; %% Paramteter C (0,inf) small C= fuck contraints, large margin big. C= use constrains, smaller margin but less mistakes in training data
% C=inf enforces no mistakes in training data, not always possible so
% output may be some random shit
max_it=10000; % max iterations
eps=0.001; % stop algorithm when error is below this value
[ w, output, value, error, cost ] = svm_grad( X,y, w0 ,C,max_it,eps);
error
marigin=2/(w'*w);
