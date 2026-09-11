% parameter recovery - analysis (use results from model recovery)
clear
clc
%close all

% general graphics, this will apply to any figure you open
% (groot is the default figure object).
set(groot, ...
    'DefaultFigureColor', 'w', ...
    'DefaultAxesLineWidth', 1.5, ...
    'DefaultAxesXColor', 'k', ...
    'DefaultAxesYColor', 'k', ...
    'DefaultAxesFontUnits', 'points', ...
    'DefaultAxesFontSize', 16, ...
    'DefaultAxesFontName', 'Helvetica', ...
    'DefaultLineLineWidth', 1, ...
    'DefaultTextFontUnits', 'Points', ...
    'DefaultTextFontSize', 14, ...
    'DefaultTextFontName', 'Helvetica', ...
    'DefaultAxesBox', 'off');

% set the tickdirs to go out - need this specific order
set(groot, 'DefaultAxesTickDir', 'out');

currentDir = pwd;
project_folder = currentDir(1:length(currentDir)-14);

expt = 4;
%models = {'DTC', 'Bayes', 'LA_DTC'};
models = {'DTC', 'Bayes'};
nRatings = 6;

if expt == 1 || expt == 4
    contrast = [1,1.33,1.78];
else
    contrast = [1,2.33,4.67,10];
end 

VBA_folder = [project_folder,'/VBA-toolbox-master'];
addpath(genpath(VBA_folder))
m0d1s1l1 = 1;

if m0d1s1l1 == 1
    nSubplot = 3;
else
    nSubplot = 4;
end



if expt == 1 || expt == 4
    nSimulation = 20;
elseif expt == 2
    nSimulation = 15;
else
    nSimulation = 11;
end

simulations = 1:nSimulation;


for model = 1:length(models) % model to recover
    clear recovered_params simulated_params
    modelName = models{model};
    figure;
    M = brewermap(6,'*Blues');
    M2 = brewermap(6,'*Reds');
    colors_blue = {M(1,:), M(2,:), M(3,:),M(4,:)};
    colors_red = {M2(1,:), M2(2,:), M2(3,:),M2(4,:)};
    colors{1} = colors_blue;
    colors{2} = colors_red;
    

    
    % Results folder
    resultsFolder = [currentDir,'/fittingResults_expt', num2str(expt),'/',modelName];
    % results file
    if m0d1s1l1 == 0
        resultsFile = [resultsFolder,'/','fittingResults_', modelName];
    else
        resultsFile = [resultsFolder,'/','fittingResults_', modelName, '_m0d1s1l1'];
    end
    load(resultsFile);
    
    % recovered params
    for sim = simulations
        recovered_params(sim,:) = params(:,sim);
    end

    % simulated params
    if m0d1s1l1 == 0
        load(['data_expt',num2str(expt),'/simulatedData_',num2str(model)]);
    else
        load(['data_expt',num2str(expt),'/simulatedData_',num2str(model), '_m0d1s1l1']);
    end

    simulated_params = simulatedParams(simulations,:);

    
    if model ~= 3

        if m0d1s1l1 == 0
            if expt == 1 || expt == 4
                mu_idx = 1:3; 
            else
                mu_idx = 1:4;
            end
    
            confCriteria_idx = mu_idx(end)+1:mu_idx(end)+11; 
            meta_noise_idx = confCriteria_idx(end)+1;
            lapse_idx = meta_noise_idx(end)+1;
        else
            mu_idx = 1;
            confCriteria_idx = 2:1+nRatings;
            lapse_idx = confCriteria_idx(end)+1;
        end

        % mu
        subplot(1,nSubplot,1);
        for cond = 1:length(mu_idx)
            mu_sim = simulated_params(:,mu_idx(cond)); 
            mu_rec = recovered_params(:,mu_idx(cond));
            scatterPlot(mu_sim,mu_rec,'Sim \mu','Rec \mu',colors{model}{cond}); 
            hold on
        end
        if model == 1
            [slopes(model,1),pvals(model,1)] = displaySlope(mu_sim(:),mu_rec(:),0.45,0.3);
        else
            [slopes(model,1),pvals(model,1)] = displaySlope(mu_sim(:),mu_rec(:),0.21,0.15);
        end
        
        % confCriteria
        subplot(1,nSubplot,2);
        confCriteria_sim = simulated_params(:,confCriteria_idx); 
        confCriteria_rec = recovered_params(:,confCriteria_idx);
        scatterPlot(confCriteria_sim(:),confCriteria_rec(:),'Sim confCriteria','Rec confCriteria',colors{model}{1}); 
        hold on
        if model == 1
            [slopes(model,2),pvals(model,2)] = displaySlope(confCriteria_sim(:),confCriteria_rec(:),3,1);
        else
            [slopes(model,2),pvals(model,2)] = displaySlope(confCriteria_sim(:),confCriteria_rec(:),.75,.6);
        end
        
        % lapse
        subplot(1,nSubplot,3)
        lapse_sim = simulated_params(:,lapse_idx);
        lapse_rec = recovered_params(:,lapse_idx);
        scatterPlot(lapse_sim(:),lapse_rec(:), 'Sim lapse','Rec lapse',colors{model}{1});
        hold on
        if model == 1
            [slopes(model,3),pvals(model,3)] = displaySlope(lapse_sim(:),lapse_rec(:),.025,.02);
        else
            [slopes(model,3),pvals(model,3)] = displaySlope(lapse_sim(:),lapse_rec(:),.12,.05);
        end
        if m0d1s1l1 == 0
            % metacognitive noise
            subplot(1,nSubplot,4)
            meta_noise_sim = simulated_params(:,meta_noise_idx);
            meta_noise_rec = recovered_params(:,meta_noise_idx);
            scatterPlot(meta_noise_sim(:),meta_noise_rec(:), 'Sim meta lapse','Rec meta lapse',colors{model}{1});
            hold on
            if model == 1
                [slopes(model,3),pvals(model,3)] = displaySlope(meta_noise_sim(:),meta_noise_rec(:),.3,.16);
            else
                [slopes(model,3),pvals(model,3)] = displaySlope(meta_noise_sim(:),meta_noise_rec(:),0.5,.3);
            end
        end

    elseif model == 3
        sigma_idx = 1; confCriteria_idx = 2:7; lapse_idx = 8;
        
        %sigma
        subplot(1,3,1)
        sigma_sim = simulated_params(:,sigma_idx);
        sigma_rec = recovered_params(:,sigma_idx);
        scatterPlot(sigma_sim(:),sigma_rec(:),'Sim \sigma','Rec \sigma',colors{1});
        hold on
        [slopes(model,1),pvals(model,1)] = displaySlope(sigma_sim(:),sigma_rec(:),.058,.02);
        
        %confCriteria
        subplot(1,3,2);
        confCriteria_sim = simulated_params(:,confCriteria_idx); 
        confCriteria_rec = recovered_params(:,confCriteria_idx);
        scatterPlot(confCriteria_sim(:),confCriteria_rec(:),'Sim confCriteria','Rec confCriteria',colors{1}); 
        hold on
        [slopes(model,2),pvals(model,2)] = displaySlope(confCriteria_sim(:),confCriteria_rec(:),1.6,-3);
        
        %lapse
        % lapse
        subplot(1,3,3)
        lapse_sim = simulated_params(:,lapse_idx);
        lapse_rec = recovered_params(:,lapse_idx);
        scatterPlot(lapse_sim(:),lapse_rec(:), 'Sim lapse','Rec lapse',colors{1});
        hold on
        [slopes(model,3),pvals(model,3)] = displaySlope(lapse_sim(:),lapse_rec(:),.058,.015);
    end
end
        
    
    
