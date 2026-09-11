function [stim, resp, conf] = simulateResponses(respProb, N)

% Simulate responses from response probabilities


stimMatrix = [zeros([1,length(respProb{1})]); ones([1,length(respProb{1})])];

respMatrix = [zeros([2,length(respProb{1})/2]), ones([2,length(respProb{1})/2])];

confScale = 1:length(respProb{1})/2
confMatrix = [repmat(fliplr(confScale),2,1), repmat(confScale,2,1)];

nConditions = length(respProb); 

for condition = 1:nConditions
    respCounts = round(respProb{condition}.*N);

    for current = 1:numel(respProb{condition})    
        if current == 1
            stim{condition}(1:respCounts(current)) = stimMatrix(current).*ones([1,respCounts(current)]);
            resp{condition}(1:respCounts(current)) = respMatrix(current).*ones([1,respCounts(current)]);
            conf{condition}(1:respCounts(current)) = confMatrix(current).*ones([1,respCounts(current)]);

        else
            stim{condition}(end+1:end+respCounts(current)) = stimMatrix(current).*ones([1,respCounts(current)]);
            resp{condition}(end+1:end+respCounts(current)) = respMatrix(current).*ones([1,respCounts(current)]);
            conf{condition}(end+1:end+respCounts(current)) = confMatrix(current).*ones([1,respCounts(current)]);
        end
    end
end

    

