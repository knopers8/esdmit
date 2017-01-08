current_folder=pwd;
data_folder=uigetdir;
cd(data_folder);
folders=dir('*new*')
fileID = fopen('results.txt','w');
for i=1:length(folders)
    name=folders(i).name;
    fprintf(fileID,'%s\n',name);
    cd(strcat(data_folder,'\',name))
    result = importdata('classified_outputs.txt');
    label  = importdata('correct_outputs.txt');
    good_percentage=100*(sum((result-label)==0) /length(result));
    heathy_id=round(mean(label));
    fprintf(fileID,'Percentage of good classifications %f\n',good_percentage);
    fprintf(fileID,' \n');
end
fclose(fileID)
cd(data_folder)