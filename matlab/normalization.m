function [ data, means, stds ] = normalization( data )
    means=zeros(size(data,2),1);
    stds=zeros(size(data,2),1);
    for j = 1:size(data,2)
        vec = data(:,j);
        means(j)=mean(vec);
        vec = vec - means(j);
        stds(j)=std(vec);
        vec = vec/stds(j);
        data(:,j) = vec;
    end

end

