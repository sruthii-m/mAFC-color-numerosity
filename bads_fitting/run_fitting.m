%run_fitting

clear
clc

% Define global variable (to use in logL_func)
global data_sub expt

% Select which experiment to fit data
expt = 3; %1, 2, 3

% Load the data for modeling
load (['dataForModeling_Expt', num2str(expt), '.mat']);


% Perform fit several times
for fit_num=1
    disp('--------------------')
    disp(['FITTING PROCEDURE NUMBER: ' num2str(fit_num)])
    disp('--------------------')
    
    
    if exist(['fittingResults/Expt' num2str(expt) 'fit_' num2str(fit_num) '.mat'], 'file')   
        load(['fittingResults/Expt' num2str(expt) 'fit_' num2str(fit_num) '.mat']);
        start_subj = length(logL) + 1;
    else 
        start_subj = 1;
    end
    
    % Loop over all subjects
    for sub = start_subj:length(data)
        disp('----------')
        disp(['Fitting subject ' num2str(sub)])
        disp('----------')
                    
        % Make the data visible to all functions
        data_sub.data = data{sub};
        data_sub.info = info;
        
        % Fit the model and save the results
        [params(:,sub), logL(sub), modelFit{sub}] = fitOneSub;
        save(['fittingResults/Expt' num2str(expt) '/fit_' num2str(fit_num)], 'params', 'logL', 'modelFit');
    end
end