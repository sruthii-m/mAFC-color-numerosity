function [slope,pval] = displaySlope(x,y,xPos,yPos)
% Perform linear regression and print slope & p-val at specified location
% in the plot

[b,stats] = robustfit(x,y);
slope = b(2); pval = stats.p(2);
text(xPos,yPos,['slope = ',num2str(round(slope,2))],'fontsize',14)

end

