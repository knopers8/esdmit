function [ classificator, output,error] = train_multi_svm( X,y, w0 ,C,max_it,eps,kernel)
%% INPUTS
% X - training data, [N,m]- m-number of dimensions, N- number of data
% points
% y - training data classification. Sequence of integers 1:d is suggested
% for labeling classes
% w0 - starting vector.
% C, max_it, eps - params described in main

 [n,~]=size(y); 
 classes=unique(y);
 m=length(classes);% number of classes
 labels=zeros(n,m);
 
 
 output=zeros(n,1);
 cl_value=zeros(n,m);
 classificator=zeros(length(w0),m);
 for i=1:m % all clas iterations
    labels(:,i)= 2*(y==classes(i))-1; % 1 if point belong to i-th class, -1 if not
     [classificator(:,i) ,~ , cl_value(:,i),~ , ~ ] = train_svm_grad( X,labels(:,i), w0 ,C,max_it,eps,kernel);
 end
 for i=1:n % all points iterations
    [~,ind] = max((cl_value(i,:)));
     output(i)=classes(ind);
 end
 error=~(output==y); % error, 1 if output differs from input class
end

