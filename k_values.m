function [l_values, k_values, mean_image] = k_values(subject_indices, data_trn, U, N)

% moyenne
mean_image = mean(data_trn, 2);

% Coposants l
l_values = [1, 2, 4, 8, 16, 32, 60];

total_subplots = length(subject_indices) * length(l_values);


% Centrer l'image
data_trn_centered = data_trn - mean_image * ones(1, size(data_trn, 2));

total_variance = sum(var(data_trn, 0, 2));
k_values = zeros(length(l_values), 1);

% Loop over the number of components to calculate k(l)
for i = 1:length(l_values)
    l = l_values(i);
    
    % Projection
    projections = U(:, 1:l) * (U(:, 1:l)' * data_trn_centered);
    
    numerator = sum(sum(projections .^ 2, 1), 2) / N;
    
    % k(l)
    k_values(i) = numerator / total_variance;
end

