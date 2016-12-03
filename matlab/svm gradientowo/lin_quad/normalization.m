function [ data ] = normalization( data )
    for j = 1:size(data,2)
        vec = data(:,j);
        vec = vec - mean(vec);
        vec = vec/std(vec);
        data(:,j) = vec;
    end

end

