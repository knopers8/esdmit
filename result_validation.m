% Script calulates results of C++ SVM classyfication and saves them in txt
% file 'results.txt' when asked for directory guide program to
% \tests_results folder.

current_folder=pwd;
data_folder=uigetdir;
cd(data_folder);
folders=dir('*unique*')  % Place to change keyword to separate results you want to use
fileID = fopen('results.txt','w');
for i=1:length(folders)
    name=folders(i).name;
    fprintf(fileID,'%s\n',name);
    cd(strcat(data_folder,'\',name))
    result = importdata('classified_outputs.txt');
    label  = importdata('correct_outputs.txt');
    time  = importdata('times.txt');
    good_percentage=100*(sum((result-label)==0) /length(result));
    fprintf(fileID,'Percentage of good classifications %f\n',good_percentage);
    
     healthy_id=mode(label);
     healthy_result=result== healthy_id;
     sick_result=result~= healthy_id;
     healthy_label=label== healthy_id;
     sick_label=label~= healthy_id;
     
     TP =sum(sick_result==sick_label) ;%True positive
     TN =sum(healthy_result==healthy_label) ;% True negative
     FP =sum(and(sick_result,healthy_label));  % False positive
     FN  =sum(and(healthy_result,sick_label)); % False negative
     sensivity=TP/(TP+FN);
     specificity=TN/(TN+FP);
     fprintf(fileID,'Sensivity %f\n',sensivity);
     fprintf(fileID,'Specificity %f\n',specificity);
      fprintf(fileID,'Full running time:%.2f [ms]   training time: %.2f [ms]   classyfication time: %.2f  [ms] \n',time);
    fprintf(fileID,' \n');
end
fclose(fileID)
cd(data_folder)