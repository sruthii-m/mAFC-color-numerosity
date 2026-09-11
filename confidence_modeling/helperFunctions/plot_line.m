function plot_line(nCondition, Y, color, xtext, ytext, plot_error_bar ,marker)

%subjects = 1:size(Y,3);
%conf_cond_config = nanmean(Y(:,:,subjects),3);
conf_cond_config = nanmean(Y,2);

for cond = 1:size(Y,1) 
    %SEM(cond) = std(conf_cond_config(cond,:))/sqrt(size(conf_cond_config,2));
    conf_cond_config_each_cond = conf_cond_config(cond,:,:);
    %SEM(cond) = nanstd(conf_cond_config(cond,:,:))/sqrt(length(conf_cond_config(cond,:,:)));
    SEM(cond) = nanstd(conf_cond_config(cond,:,:))/...
        sqrt(length(conf_cond_config_each_cond(~isnan(conf_cond_config_each_cond))));
end
plot(1:nCondition,nanmean(conf_cond_config,3)','-o',...
        'linewidth', 2, 'color', color,'LineWidth',2.5,'Marker',marker, 'MarkerSize', 9);
hold on
if plot_error_bar == 1
    errorbar(1:nCondition,nanmean(conf_cond_config,3)', SEM,'k','linestyle','none','LineWidth', 1);
    %errorbar(1:nCondition,nanmean(conf_cond_config,2)', SEM,'k','linestyle','none','LineWidth', 1);
end
xlabel(xtext, 'FontSize', 16);
ylabel(ytext,'FontSize', 16);
box off