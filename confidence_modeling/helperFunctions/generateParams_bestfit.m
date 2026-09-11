function [sim_params] = generateParams_bestfit(nConditions,nSims,params)
% Take the best fit parameter from best fitted model
% params - mu(nConditions), c(1), metaNoise(1), confCriteria(nratings-1)

global modelToFit

if modelToFit.sym_crit == 0
    nCriteria = 11;
else
    nCriteria = 6;
end

for sim = 1:nSims
    if modelToFit.dprime_prop_contr == 0
        mu(sim,:) = params(1:nConditions,sim)';
        criteria(sim,:) = params(nConditions+1:nConditions+nCriteria,sim)';
    else
        mu(sim,:) = params(1,sim)';
        criteria(sim,:) = params(2:1+nCriteria,sim)';
    end

    if modelToFit.lapse == 1
        lapse(sim,1) = params(end,sim)';
        if modelToFit.meta_noise == 1
            meta_noise(sim,1) = params(end-1,sim)';
        end
    else
        if modelToFit.meta_noise == 1
            meta_noise(sim,1) = params(end,sim)';
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





