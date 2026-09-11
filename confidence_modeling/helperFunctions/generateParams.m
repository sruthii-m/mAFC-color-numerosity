function [sim_params] = generateParams(nConditions,nSims,params)
% Generate parameters for model recovery
% Randomly generate parameters for a given model (all parameters are
% randomly generated within a range for model recovery)
% params - mu(nConditions), c(1), metaNoise(1), confCriteria(nratings-1)

global modelToFit

nCriteria = 11;
for sim = 1:nSims
    
    % Sample confidence criteria from the fitted paramters
    % If it's Bayesian model, thresholds are sampled
    for crit = 1:nCriteria
        crit_lowerLim(crit) = min(params(crit+nConditions,:));
        crit_upperLim(crit) = max(params(crit+nConditions,:));
        crit_range(crit) = crit_upperLim(crit) - crit_lowerLim(crit);
        criteria(sim,crit) = crit_lowerLim(crit) + crit_range(crit) * rand(1);
        
        if crit > 1
            while criteria(sim,crit) <= criteria(sim,crit-1)
                criteria(sim,crit) = crit_lowerLim(crit) + crit_range(crit) * rand(1);
            end
        end
    end
    
    for cond = 1:nConditions
        mu_lowerLim(cond) = min(params(cond,:));
        mu_upperLim(cond) = max(params(cond,:));
        mu_range(cond) = mu_upperLim(cond) - mu_lowerLim(cond);
        mu(sim,cond) = mu_lowerLim(cond) + mu_range(cond) * rand(1);
        
        if cond > 1
            while mu(sim,cond) <= mu(sim,cond-1)
                mu(sim,cond) = mu_lowerLim(cond) + mu_range(cond) * rand(1);
            end
        end
    end
    
    if modelToFit.lapse == 1
        lapse(sim,1) = min(params(end,:)) + (max(params(end,:))-min(params(end,:))) * rand(1);
        if modelToFit.meta_noise == 1
            meta_noise(sim,1) = min(params(end-1,:)) + (max(params(end-1,:))-min(params(end-1,:))) * rand(1);
        end
    else
        if modelToFit.meta_noise == 1
            meta_noise(sim,1) = min(params(end,:)) + (max(params(end,:))-min(params(end,:))) * rand(1);
        end   
    end
    
end


if modelToFit.lapse == 1  
    if modelToFit.meta_noise == 1
        sim_params = [mu, criteria, meta_noise, lapse];
    else
        sim_params = [mu,criteria,lapse];
    end
else
    if modelToFit.meta_noise == 1
        sim_params = [mu,criteria,meta_noise];
    else
        sim_params = [mu,criteria];
    end
end
end

