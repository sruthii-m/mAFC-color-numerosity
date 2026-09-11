function [h] = scatterPlot(x,y,xlab,ylab,color)
%Scatter plots of model and experimental data

%axes('box','off','tickdir','out','LineWidth',1.25,'FontSize',13); hold on;

axis square; 
hold on; set(gca,'fontsize',16,'box','off','tickdir','out','linewidth',1.5)
h = plot(x,y,'.','markersize',30,'Color',color,'LineWidth',2); hold on;
plot(xlim,xlim,'--k','LineWidth',1)

ylabel(ylab)
xlabel(xlab)

end

