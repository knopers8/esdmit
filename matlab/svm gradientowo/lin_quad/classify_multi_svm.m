function [output, value] = classify_multi_svm( X, classificator,kernel)
%% INPUTS
   N=length(X);
   m=size(classificator,2);
   value=zeros(m,N);
   output=zeros(N,1);
   for i=1:m
       for j=1:N
            X_tmp=kernel(X(j,:),classificator(end,i));
            value(i,j)=X_tmp*classificator(:,i);
       end
   end
   for i=1:N % all points iterations
         [~,ind] = max((value(:,i)));
         output(i)=ind;
   end
end

