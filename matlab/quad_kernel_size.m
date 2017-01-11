function [ D] = quad_kernel_size( d )
%QUAD_KERNEL_SIZE computes size of quadratic kernel where d is linear
%kernel eg.:
%- linear = [w1 w2], d=2
%- quadratic= [w1*w1 sqrt(2)w1*w2 w2*w2] D=3
% Note: we are using [w1 w2 b] so above example don't match
% https://wikimedia.org/api/rest_v1/media/math/render/svg/7a8a37a96bc8bce31c758d7c378cb2b46db6ece3
%     D=2*d+1; % first and last sum of the elements + c
    D=d; % first and last sum of the elements + c
    for i=2:d % This double sum in middle
        for j=1:i-1
            D=D+1;            
        end       
    end        

end

