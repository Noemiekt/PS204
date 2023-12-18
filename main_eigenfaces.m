% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;
dbstop if error;


%% Data extraction
% Training set
adr = './database/training1/';
fld = dir(adr);
nb_elt = length(fld);

% Data matrix containing the training images in its columns 
data_trn = []; 

% Vector containing the class of each training image
lb_trn = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)];
    end
end

% Size of the training set
[P,N] = size(data_trn);

% Calcul de U
[U, Nc, size_cls_trn] = eigenfaces(data_trn, lb_trn, P, N);
% Affichage de la database
F = zeros(192*Nc,168*max(size_cls_trn));
for i=1:Nc
    for j=1:size_cls_trn(i)
          pos = sum(size_cls_trn(1:i-1))+j;
          F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
    end
end
figure;
imagesc(F);
colormap(gray);
axis off;

%% Affichage des valeurs propres
Ne = size(U, 2); 
Nr = ceil(sqrt(Ne));
figure;
for i = 1:Ne
    subplot(Nr, Nr, i);
    eigenface = reshape(U(:, i), [192, 168]); 
    imshow(eigenface, []);
    title(['Eigenface ', num2str(i)]);
end
sgtitle('All Eigenfaces');

%% Calcul des k

% Indices des sujets
subject_indices = [1, 11, 21, 31, 41, 51]; 

[l_values, k_values, mean_image] = k_values(subject_indices, data_trn, U, N);

figure;

for j = 1:length(subject_indices)
    subject_idx = subject_indices(j);
    original_image = data_trn(:, subject_idx);

   for i = 1:length(l_values)
        
        % Pour l'affichage
        subplot_index = (j - 1) * length(l_values) + i;
        l = l_values(i);
        
        % Projection
        projection = U(:, 1:l)' * (original_image - mean_image);
        
        % Reconstruction
        reconstruction = U(:, 1:l) * projection;
        reconstruction = reconstruction + mean_image;
        
        % Affichage
        subplot(length(subject_indices), length(l_values), subplot_index);
        imshow(reshape(reconstruction, [192, 168]), []);
        title(['Subject : ', num2str(subject_idx), ', ',  ' l :', num2str(l)]);
    end
end

sgtitle('Reconstruction Evolution for Each Subject');

% Plot k(l)
figure;
plot(l_values, k_values, '-o');
title('Reconstruction Ratio κ(l)');
xlabel('Number of Components l');
ylabel('κ(l)');



%% Extraction des l_[phi] = classifieur(x,S,Bx,k,m), [phi] = classifieur(x,S,Bx,k,m) star premières colonnes de la matrice U

S = U(:, 1:l_values(l_star-1));
x = data_trn(:,11);
Bx = data_trn;
k = 12;
m = 10; % nobre d'individues presents dans une classe

phi = classifieur(x,S,Bx,k,m);




