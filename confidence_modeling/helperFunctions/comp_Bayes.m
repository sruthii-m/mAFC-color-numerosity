function conf = comp_Bayes(signal,conf_crit,a,N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The model gives confidence based on a Bayesian computation of the 
% probability that the choice is correct. The algorithm uses numerical
% approximations. It computes the likelihood that a specific number of dots 
% (300 options are chosen spanning the relevant interval) was presented for 
% each color on each simulated trial. Then, based on these values,
% posterior probability of being correct is computed in each case, and then
% used for determining confidence.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Set parameters for numerical approximation
%max_value_to_test = 200; %test true activations between 0 and max_value_to_test

% Change from the attempt5: 
max_value_to_test = 110;
max_signals = max(signal);
signal = (signal ./ max_signals) * 100; %scale the signal so that highest signal is 100

likelihood = zeros(max_value_to_test+1,size(signal,1),N);
values_to_test = repmat([1:max_value_to_test]',1,N);

% Compute the likelihood of each signal observation for each value in the range
% (Note that in each iteration, we are performing max_value_to_test * N independent computations, 
% such that X, MU, and SIGMA in normpdf are matrices of dimension max_value_to_test * N.
% The likelihoods for a true activation of 0 are computed separately.)
for color=1:size(signal,1)
    likelihood(1,color,:) = signal(color,:)==0;
    likelihood(2:max_value_to_test+1,color,:) = normpdf(repmat(signal(color,:),max_value_to_test,1),values_to_test,a*values_to_test);
    %likelihood(2:max_value_to_test+1,color,:) = normpdf(repmat(signal(color,:),max_value_to_test,1),values_to_test,qqrt(a*values_to_test));
end

% Normalize the likelihoods into a probability that each of the value_to_test
% was the true number of dots (assuming a uniform prior)
prob_value_to_test = likelihood ./ sum(likelihood);

% Compute the cumulative probability distributions
cum_prob_value_to_test_cur = cumsum(prob_value_to_test);

% Insert one row in the beginning for the ease of coding later
cum_prob_value_to_test = [zeros(1,size(cum_prob_value_to_test_cur,2),size(cum_prob_value_to_test_cur,3));cum_prob_value_to_test_cur];

% Loop over the colors and compute the posterior probability of being
% correct for each option
post_prob = zeros(size(signal,1),N);

%% You will need 2 versions of this - one for 3 choices, one for 4 choices
for color1=1:size(signal,1)
    n = size(cum_prob_value_to_test,1);
    if size(signal,1) == 3
        % Determine the numbers of the alternative colors
        color2 = rem(color1+1,3);  if color2==0, color2=3; end
        color3 = rem(color1+2,3);  if color3==0, color3=3; end
        
        % Compute the posterior probability for each color by multiplying the probability of a specific number of dots with
        % the probability that the remaining colors have fewer dots. Correct for the case of equal number of dots.
        % Probability of A being largest = P(A>B,C) + .5*P(A=B>C) + .5*P(A=C>B) + 1/3*P(A=B=C)
        PA1_larger_than_rest = sum(prob_value_to_test(:,color1,:) .* cum_prob_value_to_test(1:n-1,color2,:) .* cum_prob_value_to_test(1:n-1,color3,:));

        P_2prob_equal = .5*sum(prob_value_to_test(:,color1,:) .* prob_value_to_test(:,color2,:) .* cum_prob_value_to_test(1:n-1,color3,:)) + ...
            .5*sum(prob_value_to_test(:,color1,:) .* cum_prob_value_to_test(1:n-1,color2,:) .* prob_value_to_test(:,color3,:));

        P_3prob_equal = (1/3)* sum(prob_value_to_test(:,color1,:) .* prob_value_to_test(:,color2,:) .* prob_value_to_test(:,color3,:));

        post_prob(color1,:) = PA1_larger_than_rest + P_2prob_equal + P_3prob_equal;
    
    elseif size(signal,1) == 4 % For expt 1

        % Determine the numbers of the alternative colors
        color2 = mod(color1,4) + 1; 
        color3 = mod(color1 + 1,4) + 1; 
        color4 = mod(color1 + 2,4) + 1;
        
        %NEW VERSION that relies on cumulative probabilities being computed up
        %to k-1 rather than up to k For the cum_prob_value_to_test variable, a 
        % row of zeros was added on top to simplify the coding

        % Probability of A1 being largest = P(A1>A2,A3,A4)+ .5*(P(A1=A2>A3,A4)+P(A1=A3>A2,A4)+P(A1=A4>A3,A2))+
        % 1/3*(P(A1=A2=A3>A4)+P(A1=A2=A4>A3)+P(A1=A3=A4>A2))+1/4*P(A1=A2=A3=A4)
        PA1_larger_than_rest = sum(prob_value_to_test(:,color1,:) .* cum_prob_value_to_test(1:n-1,color2,:) .*...
            cum_prob_value_to_test(1:n-1,color3,:) .* cum_prob_value_to_test(1:n-1,color4,:));

        P_2prob_equal = (1/2)*(sum(prob_value_to_test(:,color1,:) .* prob_value_to_test(:,color2,:) .* cum_prob_value_to_test(1:n-1,color3,:).* cum_prob_value_to_test(1:n-1,color4,:)) + ...
            sum(prob_value_to_test(:,color1,:) .* cum_prob_value_to_test(1:n-1,color2,:) .* prob_value_to_test(:,color3,:) .* cum_prob_value_to_test(1:n-1,color4,:)) + ...
            sum(prob_value_to_test(:,color1,:) .* prob_value_to_test(:,color4,:) .* cum_prob_value_to_test(1:n-1,color3,:) .* cum_prob_value_to_test(1:n-1,color2,:)));

        P_3prob_equal = (1/3)*(sum(prob_value_to_test(:,color1,:) .* prob_value_to_test(:,color3,:) .* prob_value_to_test(:,color4,:).* cum_prob_value_to_test(1:n-1,color2,:)) + ...
            sum(prob_value_to_test(:,color1,:) .* prob_value_to_test(:,color2,:) .* prob_value_to_test(:,color4,:).* cum_prob_value_to_test(1:n-1,color3,:)) + ...
            sum(prob_value_to_test(:,color1,:) .* prob_value_to_test(:,color2,:) .* prob_value_to_test(:,color3,:).* cum_prob_value_to_test(1:n-1,color4,:)));

        P_4prob_equal = (1/4)* sum(prob_value_to_test(:,color1,:) .* prob_value_to_test(:,color2,:) .* prob_value_to_test(:,color3,:).* prob_value_to_test(:,color4,:));

        post_prob(color1,:) = PA1_larger_than_rest + P_2prob_equal + P_3prob_equal + P_4prob_equal;
    end
end

% Determine the response and confidence variable (the posterior probability of the chosen stimulus)
conf_variable = max(post_prob);

% Determine the confidence judgment
conf = ones(1,N);
conf(conf_variable > conf_crit(1)) = 2;
conf(conf_variable > conf_crit(2)) = 3;
conf(conf_variable > conf_crit(3)) = 4;
