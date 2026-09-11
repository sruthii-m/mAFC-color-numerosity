%% Generate data for model recovery for mAFC
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate datasets using the best fitting parameters from the best-fitted 
% 4 parameter decision model and the best fitted confidence model
%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
clc

% models to simulate
models = {'Bayes','Diff','PE'};
% Decide which dataset to simulate
expt = 2;
% nratings for the 3 experiments of mAFC are all 4
nratings = 4; 

% Set the fixed parameters
if expt == 1
    mu = [100, 90, 85, 75, 65, 50, 40, 30, 25, 20, 15, 0];
    mu_number = [1, 3, 12, 12;...  % p.cond{1} = [100,85,0,0];
        1, 2, 12, 12; ...          % p.cond{2} = [100,90,0,0];
        1, 3, 8, 12;...            % p.cond{3} = [100,85,30,0];
        1, 4, 7, 12;...            % p.cond{4} = [100,75,40,0];
        1, 5, 6, 12;...            % p.cond{5} = [100,65,50,0];   
        1, 3, 11, 11;...           % p.cond{6} = [100,85,15,15];
        1, 4, 10, 10;...           % p.cond{7} = [100,75,20,20];
        1, 5, 9, 9];               % p.cond{8} = [100,65,25,25];
    N = 576; % number of trials in the original experiment
    nSims = 24; % number of subjects in the original experiment
elseif expt == 2
    mu = [100, 80, 75, 60, 40, 32, 0];
    mu_number = [1, 3, 7;...  % p.cond{1} = [100,75,0];
        2, 4, 7;...           % p.cond{2} = [80,60,0];
        1, 3, 5;...           % p.cond{3} = [100,75,40];
        1, 3, 3;...           % p.cond{4} = [100,75,75];
        2, 4, 6;...           % p.cond{5} = [80,60,32]; 
        2, 4, 4];             % p.cond{6} = [80,60,60];
    N = 576; % number of trials in the original experiment
    nSims = 25; % number of subjects in the original experiment
elseif expt == 3
    mu = [98, 84, 72, 62, 60, 52, 48, 42];
    mu_number = [1, 2, 3; ... % p.cond{1} = [98,84,72];
        1, 2, 5; ...        % p.cond{2} = [98,84,60];
        1, 2, 7; ...        % p.cond{3} = [98,84,48];
        1, 3, 3; ...        % p.cond{4} = [98,72,72];
        1, 3, 5; ...        % p.cond{5} = [98,72,60];
        1, 3, 7; ...        % p.cond{6} = [98,72,48];
        2, 3, 4; ...        % p.cond{7} = [84,72,62];
        2, 3, 6; ...        % p.cond{8} = [84,72,52];
        2, 3, 8; ...        % p.cond{9} = [84,72,42];
        2, 4, 4; ...        % p.cond{10} = [84,62,62];
        2, 4, 6; ...        % p.cond{11} = [84,62,52];
        2, 4, 8];           % p.cond{12} = [84,62,42];
    N = 1440; % number of trials in the original experiment
    nSims = 15; % number of subjects in the original experiment
end

% Get current folder and add helper functions
currentDir = pwd;
project_folder = currentDir(1:length(currentDir)-14); % folder for conf models
main_project_folder = project_folder(1:end-11); % folder for conf and decision models
helper_func_folder = [project_folder, '/helperFunctions'];
fittingResultsFolder = [project_folder, '/fittingResults'];
addpath(helper_func_folder);

% Load experimental data for the info
load([project_folder '/dataForModeling_Expt' num2str(expt) '.mat'])
nChoice = size(mu_number,2);
[nCond,nConfig] = size(data{1}.resp);

% Load the best decision model fitting results using 4 params fit
load([main_project_folder '/4params_fit/fittingResults/Expt' num2str(expt) '/fit_best.mat']);
dec_params = params; % follows the order of a, color bias (3 for expt1, 2 for expt2&3, lapse)


for model = 1:length(models)

    disp('------------------------------')
    disp(['Model: ' models{model} ])
    disp('------------------------------')


    clear simulated_data params conf_params
    % Load the best confidence model fitting results
    load([fittingResultsFolder '/Expt' num2str(expt) '/' models{model} '_best']);
    conf_params = params; % 3 confidence criteria
    
    % loop over all simulated subjects
    for sim = 1:nSims
    
        % the multiplier for color bias, the first color bias is set to be 1
        multipliers = [1,dec_params(2:nChoice,sim)'];
    
        for cond = 1:nCond
            for config = 1:nConfig
                % Determine the relevant mus and sigmas
                relevant_mu_numbers = mu_number(cond, info.configs(config,:));
                mus_condConfig = mu(relevant_mu_numbers) .* multipliers;
                sigmas_condConfig = dec_params(1,sim) * mus_condConfig; %equation is variance_N = a*N;
                %sigmas_condConfig = sqrt(variance_condConfig); % get the sigma of the sensory distribution
                % Simulate the activations for each color
                for color=1:nChoice
                    signal(color,:) = normrnd(mus_condConfig(color), sigmas_condConfig(color), 1, N/(nCond*nConfig));
                end
    
                % Determine the responses in the absence of a lapse
                [~, resp_model] = max(signal);
    
                % Compute confidence responses
                if strcmp(models{model},'Bayes')
                    conf_model = comp_Bayes(signal,conf_params(:,sim),dec_params(1,sim),N/(nCond*nConfig));
                else
                    conf_model = eval(['comp_' models{model} '(signal,conf_params(:,sim),N/(nCond*nConfig))']);
                end
                simulated_data{sim}.resp{cond,config} = resp_model;
                simulated_data{sim}.conf{cond,config} = conf_model;
            end
        end
    
        % add the lapse rate
        % the last decision parameter is the lapse rate
        lapse = dec_params(end,sim);
        % generate random resp and conf responses
        rand_resp = randi(nChoice,1,floor(lapse * N));
        rand_conf = randi(nratings,1,floor(lapse * N));
    
        % determine random cell numbers for the lapse trials
        cell_number = randi(nCond*nConfig,1,floor(lapse * N));
        array_loc_number = randi(N/(nCond*nConfig),1,floor(lapse * N));
        for trial = 1:floor(lapse * N)
            simulated_data{sim}.resp{cell_number(trial)}(array_loc_number(trial)) = rand_resp(trial);
            simulated_data{sim}.conf{cell_number(trial)}(array_loc_number(trial)) = rand_conf(trial);
        end
    
        % the commented out section below is another way to add lapse rate
        % % determine which trials are lapse rate
        % lapse_trials = randi(N,1,floor(lapse * N));
        % % find the cell number of the random trials
        % for trial = 1:length(lapse_trials)
        %     if mod(lapse_trials(trials),N/(nCond*nConfig)) == 0
        %         cell_number = floor(lapse_trials/(N/(nCond*nConfig)));
        %     else
        %         cell_number = floor(lapse_trials/(N/(nCond*nConfig)))+1;
        %     end
        % end
        % 
        % % find the location of the random trial within the array
        % loc_within_cell = mod(lapse_trials,N/(nCond*nConfig));
        % loc_within_cell(loc_within_cell == 0) = N/(nCond*nConfig);
        % 
        % for trial = 1:length(cell_number)
        %     simulated_data{sim}.resp{cell_number(trial)}(loc_within_cell(trial)) = rand_resp(trial);
        %     simulated_data{sim}.conf{cell_number(trial)}(loc_within_cell(trial)) = rand_conf(trial);
        % end
    
    end
    
    % Save simulated data
    save(['data_expt' num2str(expt) '/simulatedData_' models{model}], 'simulated_data','info')
end




