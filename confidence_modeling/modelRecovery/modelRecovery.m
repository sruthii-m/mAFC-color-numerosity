%% Run model recovery

clear
clear global
clc

global data_sub modelToFit expt dec_params

% Get current folder and add helper functions
currentDir = pwd;
project_folder = currentDir(1:length(currentDir)-14); % folder for conf models
main_project_folder = project_folder(1:end-11); % folder for conf and decision models
helper_func_folder = [project_folder, '/helperFunctions'];
fittingResultsFolder = [currentDir, '/fittingResults'];
addpath(helper_func_folder);

models =  {'Bayes','Diff','PE'};
expt = 2; % Decide which dataset to run the model recovery
True_model = 'Bayes'; % Model that the dataset was simulated from
modelToFit = 'Bayes'; % Model to fit on

% load the simulated data
dataFile = [cd,'/data_expt', num2str(expt),'/simulatedData_', True_model, '.mat'];
load(dataFile)

% Folder to save fitting results
resultsFolder = ['fittingResults_expt',num2str(expt),'/',True_model];

% Load the best decision model fitting results using 4 params fit
load([main_project_folder '/4params_fit/fittingResults/Expt' num2str(expt) '/fit_best.mat']);
decision_params_all_sub = params; % follows the order of a, color bias (3 for expt1, 2 for expt2&3, lapse)
clear params

% Loop over all (remaining) subjects
for sub=1:length(simulated_data)

    disp('--------------------------------------')
    disp(['Subject:' num2str(sub) '    Model to fit:' modelToFit '    True Model:' True_model])
    disp('--------------------------------------')

    data_sub.data = simulated_data{sub};
    data_sub.info = info;
    dec_params = decision_params_all_sub(:,sub);

    resultsFileName = [resultsFolder,'/fittingResults_',modelToFit,'.mat'];

    % Fit the model and save the results
    [params(:,sub), logL(sub), modelFit{sub}] = fitOneSub();
    save(string(resultsFileName), 'modelFit', 'params', 'logL');
end



