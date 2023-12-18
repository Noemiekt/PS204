function [l_values, k_values, mean_image, l_star] = k_values(subject_indices, data_trn, U, N)

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

% Find the smallest l that gives a k(l) of at least 0.9
l_star = find(k_values >= 0.9, 1, 'first');
if isempty(l_star)
    disp('No dimension l* found that achieves the ratio of 0.9.');
else
    disp(['The smallest dimension l* to achieve a ratio of 0.9 is: ', num2str(l_values(l_star-1))]);
end