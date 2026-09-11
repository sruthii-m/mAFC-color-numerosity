function conf = comp_PE(signal,conf_crit,N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The model gives confidence based on the strength of the highest 
% activation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Determine the signal for confidence
conf_variable = max(signal);

% Determine the confidence judgment
conf = ones(1,N);
conf(conf_variable > conf_crit(1)) = 2;
conf(conf_variable > conf_crit(2)) = 3;
conf(conf_variable > conf_crit(3)) = 4;