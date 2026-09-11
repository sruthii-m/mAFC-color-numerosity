%plot_illustration
%% Plot the illustration for signal detection theory in 3 choice task
x = linspace(-1,5);
y1 = normpdf(x,0.5,0.4);

y2 = normpdf(x,1,0.5);
y3 = normpdf(x,1.5,0.8);
y4 = normpdf(x,1.7,1);

figure('Color','w', 'DefaultAxesFontSize',14,'Position',[1341 1371 705 187]);
%plot(x, y1, 'Linewidth', 4, 'color', [0.2 0.2 0.2]);

plot(x, y2, 'Linewidth', 4, 'color', [150 46 45]/255);
hold on
plot(x, y3, 'Linewidth', 4, 'color', [97 141 179]/255);

plot(x, y4, 'Linewidth', 4, 'color', [107 131 95]/255);
hold off

set(gca,'XTick',[],'xticklabel',{[]})
h = gca;
h.YAxis.Visible = 'off';
h.XAxis.Visible = 'off';
box off

%% Plot the bar graphs for internal activations
clr = [150 46 45; 97 141 179; 107 131 95]/255;
figure('Color','w', 'DefaultAxesFontSize',14,'Position',[912 1312 316 246]);
hypo_activations = [100 60 50];
b = bar(hypo_activations,'facecolor', 'flat');
b.CData = clr;
box off
ylim([0 120])
set(gca,'XTick',[],'xticklabel',{[]})
set(gca,'YTick',[],'yticklabel',{[]})
h = gca;
