function [ data] = classificator_normalization( data , means, stds )
    
    for j = 1:size(data,2)
        vec = data(:,j);
        vec = vec - means(j);
        vec = vec/stds(j);
        data(:,j) = vec;
    end

end

