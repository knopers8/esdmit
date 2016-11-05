%% initialization and configuration

clear
close all;

% most of these records are invalid - number of inputs and outputs are
% different.
records = [100   101   102   103   104   105   106   107   108   109 ...
   111   112   113   114   115   117   118   119   121   122 ...
   123   124   200   201   202   203   205   208   209   210 ...
   212   213   214   215   217   219   220   221   222   223   228 ...
   230   231   232   233   234];

% records = [100];

normalization_enabled = true;




%% load data

all_data = [];
all_class_id = [];

for i = 1:length(records)
    clear data;
    clear class_id;
    
    file_path = ['ReferencyjneDane/', num2str(records(i)), '/ConvertedQRSRawData.txt'];
    data = import_data( file_path );

    if normalization_enabled
        for j = 1:size(data,2)
            vec = data(:,j);
            vec = vec - mean(vec);
            vec = vec/std(vec);
            data(:,j) = vec;
        end
    end
    
    
    file_path = ['ReferencyjneDane/', num2str(records(i)), '/Class_IDs.txt'];
    class_id = import_class_id ( file_path );

    
    all_data = [all_data; data];
    all_class_id = [all_class_id; class_id];
    
%     disp('all data size');
%     size(all_data)
%     disp('all class id size');
%     size(all_class_id)
end


%% svm training



%% svm check


