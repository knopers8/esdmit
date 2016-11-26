function [ out] = cost_fun( X,y, w,C )
    % Calulates value of cost fuction to log progress or terminate algorithm
    % http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf pages 29 and 36
    [N,m]=size(X);
     lam=2/(N*C);
     regularization=(lam/2)*w(1:end-1)'*w(1:end-1);
     tmp=0;
     for i=1:N
         tmp=tmp+max(1-y(i)*(w(1:end-1)'*X(i,:)'+w(end)),0);
     end
     loss_funcion=tmp/N;
     out=regularization+ loss_funcion;
end

