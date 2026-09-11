%% Analyse model recovery results

clear
clc
%close all

%% set up path
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

addpath(genpath('VBA-toolbox-master'))

currentDir = pwd;
% the Attempt5/conf_model folder
project_folder = currentDir(1:length(currentDir)-14);
addpath([project_folder, '/helperFunctions']);
% the main modeling folder for all modeling work
modeling_main_folder = currentDir(1:length(currentDir)-34);


expt = 3; % 1-3
measureName = 'AIC'; % AIC or BIC
run_analyses = 0;

%models = {'DTC', 'Bayes', 'LA_DTC'};
models = {'Bayes', 'Diff', 'PE'};

% add VBA folder
VBA_folder = [modeling_main_folder,'/VBA-toolbox-master'];
addpath(genpath(VBA_folder))

%% run analyses
if run_analyses
    for model1 =  1:length(models) % model to recover -- the true model
        recovered_model = models{model1};
        % Results folder
        resultsFolder = [currentDir,'/fittingResults_expt',num2str(expt), '/',recovered_model];
        measure{model1} = [];
        for model2 = 1:length(models) % model fit
            fittedModel = models{model2};

            % load the results file
            resultsFile = [resultsFolder,'/', 'fittingResults_',fittedModel];
            load(resultsFile)
            nSims = length(logL);
            for sim = 1:nSims
                if strcmp(measureName,'AIC')
                    measure{model1}(sim,model2) = modelFit{1,sim}.AIC;
                elseif strcmp(measureName,'BIC')
                    measure{model1}(sim,model2) = modelFit{1,sim}.BIC;
                end
            end
        end
        
        % Bayesian model selection (random effects analysis)
        [posterior,out] = VBA_groupBMC(-measure{model1}');

        f(model1,:) = out.Ef ; % model frequencies
        EP = out.ep ; % exceedance probabilities
        PEP(model1,:) = (1-out.bor)*out.ep + out.bor/length(out.ep); % protected exceedance probabilities
        BOR(model1) = out.bor; % Bayesian omnibus risk

        % Recovery
        [~,best_model] = min(measure{model1}');

        for model2 = 1:length(models)
            confusionMatrix(model1,model2) = sum(best_model == model2)./nSims
        end
    end
    % save the model recovery results
    save(['recovery_results/','expt', num2str(expt),'_recovery_',measureName],'confusionMatrix','f','PEP','measure')

else
    % load the model recovery results if it exists
    load(['recovery_results/','expt', num2str(expt),'_recovery_',measureName],'confusionMatrix','f','PEP','measure')

end

%% plot the confusion matrix (after running both AIC & BIC analyses)
measureNames = {'AIC'};
count = 1;
figure('Color','w', 'DefaultAxesFontSize',16, 'Position', [460 446 725 252]);

load(['recovery_results/','expt', num2str(expt),'_recovery_',measureName, '.mat'],'confusionMatrix','f','PEP')

for m = 1:length(measureNames)
    measureName = measureNames{m};
    load(['recovery_results/','expt', num2str(expt),'_recovery_',measureName])

    all_measures = {'confusionMatrix','f'}
    if strcmp(measureName,'AIC')
        all_measure_names = {'AIC','p_{model} with AIC','p_{exc} with AIC'};
    else
        all_measure_names = {'BIC','p_{model} with BIC','p_{exc} with BIC'};
    end

    for k = 1:length(all_measures)
        measureToTest = all_measures{k}
        % Create a table for plotting
        true_model = repmat(models, 1, length(models))'
        fitted_model = reshape(repmat(models, length(models), 1),[],1)
        prob_bestModel = reshape(round(eval([measureToTest]),2),[],1);
        tbl = table(prob_bestModel,true_model,fitted_model);

        % Plot Figure
        subplot(1,2,count);
        h = heatmap(tbl,'fitted_model','true_model', 'ColorVariable','prob_bestModel', 'FontSize',14);
        grid(h,'off')
        M = brewermap(NaN,'Blues');
        %h.Colormap = M;
        color_matrix = [234 241 230;194 206 180;145 157 129;130 140 121;111 134 119;...
            110 128 110;101 116 98;91 110 81;55 78 61]/255;
        h.Colormap = color_matrix;
        
        h.GridVisible = 'off'; h.FontSize = 16;
        h.XDisplayData = models; h.YDisplayData = models; h.XLabel = 'Fitted Model'; h.YLabel = 'True Model';
        h.Title = all_measure_names{k}
        count = count+1
    end
end
% save figure
saveas(gcf,['recovery_expt', num2str(expt)],'epsc')

%% plot the actual AIC difference for model recovery
%colors = {[9 147 150]/255, [238 155 0]/255, [174 32 18]/255}; 
%colors = [167 160 125; 127 159 175; 163 91 89]/255;
%colors = [167 170 161; 167 170 161; 167 170 161]/255;

%colors = [178 182 182; 178 182 182; 178 182 182]/255; %the chosen gray

%colors = [178 191 195; 178 191 195; 178 191 195]/255;
%colors = [197 201 203; 197 201 203; 197 201 203]/255;

colors = [130 140 115; 130 140 115; 130 140 115]/255; %the chosen green

%colors = [55 78 61; 55 78 61; 55 78 61]/255;


figure('Color','w', 'DefaultAxesFontSize',16,'Position',[157 608 775 189]);
for model = 1:length(models)
    subplot(1,length(models),model);
    % get the measure diff values by subtracting the measure of true model 
    measure_diff = measure{model} - measure{model}(:,model);
    measure_diff_SEM = std(measure_diff)/sqrt(size(measure_diff,1));
    b = bar(mean(measure_diff));
    hold on
    b.FaceColor= 'flat';
    b.CData = colors;
    er = errorbar(1:length(models),mean(measure_diff),measure_diff_SEM);
    er.Color = [0 0 0];                            
    er.LineStyle = 'none'; 
    er.LineWidth = 1;
    set(gca,'xticklabel',models)
    ax = gca;
    ax.YLabel.String = ['\Delta', measureName];
    ax.Title.String = models{model};
    ax.XLabel.String = 'Recovered model';
    hold off
    box off
end
% save figure
saveas(gcf,['recovery_AIC_diff_expt', num2str(expt)],'epsc')



