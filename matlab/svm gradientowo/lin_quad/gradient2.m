function [ out] = gradient2( X,y, w,C,kernel)
% http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf pages 36-38
    [N,~]=size(X);
    m=length(w);
    lam=2/(N*C);
    L=zeros(N,m);
    for i=1:N
        X_tmp=kernel(X(i,:),w(end));
        if ((y(i)*(w'*X_tmp'))<1)   % if y*f(x)>=1 constrain is not violated so cost function=0 
            for j=1:m
                  L(i,j)=-y(i)*X_tmp(j);
            end 
%             L(i,m+1)=-y(i);
        end
    end
    out=lam*[w(1:end -1); 0]+ (sum(L)')/N;
end

