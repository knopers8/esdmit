function [ out] = gradient( X,y, w,C )
% http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf pages 36-38
    [N,m]=size(X);
    lam=2/(N*C);
    L=zeros(N,m+1);
    for i=1:N
        if ((y(i)*(w(1:end-1)'*X(i,:)'+w(end)))<1)   % if y*f(x)>=1 constrain is not violated so cost function=0 
            for j=1:m
                  L(i,j)=-y(i)*X(i,j);
            end 
            L(i,m+1)=-y(i);
        end
    end
    out=lam*[w(1:end -1); 0]+ (sum(L)')/N;
end

