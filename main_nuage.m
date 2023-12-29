%% KEGL Noémie, SALLMONE Armela & MONY Alexandra

clc;
clear all;
close all;


%% 4) Classification -
%% Représentation des nuages de points des couples de composantes principales 

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
[P,Ni] = size(data_trn); 

%% Processing

% Eigenface Matrix (U) & Nb of class (m)
[U, m, ~] = eigenfaces(data_trn, lb_trn, P, Ni);

% Number of individuals in each class
Nc = Ni/m;

% Determine the number of principal components.
subject_ind = [];
for i = 1:Nc:(m*Nc)+1
    subject_ind = [subject_ind, i];
end

[l_values, k_values, mean_image] = k_values(subject_ind, data_trn, U, Ni);
l_star = find(k_values >= 0.9, 1, 'first');

S = U(:, 1:l_values(l_star-1));
[~,l]=size(S);


% Determine the principal components. 
wbase=zeros(l,Ni); 

for i=1:Ni
    xi_moy = mean(data_trn(:,i));
    wbase(:,i) = (data_trn(:,i)-xi_moy).' * S;
end

% Determine the intra-class means
Mu_chap=zeros(l,m);

for i = 1:m
    % Index des colonnes du groupe actuel
    iC = (i - 1) * Nc + 1 : i * Nc;

    % Additionner les colonnes du groupe actuel
    Mu_chap(:, i) = sum(wbase(:, iC), 2)/Nc;
end


% Choix des 3 classes :
class1=1;
class2=2;
class3=3;


% Créez quatre graphiques distincts pour les paires de composantes principales
figure;
for i = 1:4
    subplot(2, 2, i);
    pc1 = i;
    pc2 = i + 1;
    
    % Tracez les coordonnées des images des trois individus
    scatter(wbase(pc1, 1:Nc), wbase(pc2, 1:Nc), 'ro', 'filled');
    hold on;
    scatter(wbase(pc1, Nc+1:2*Nc), wbase(pc2, Nc+1:2*Nc), 'bo', 'filled');
    hold on;
    scatter(wbase(pc1, 2*Nc+1:3*Nc), wbase(pc2, 2*Nc+1:3*Nc), 'go', 'filled');
    hold on;

    % Ajout de la moyenne intra-classe
    scatter(Mu_chap(pc1, class1), Mu_chap(pc2,class1), 'rx', 'LineWidth', 2);
    hold on;
    scatter(Mu_chap(pc1, class2), Mu_chap(pc2, class2), 'bx', 'LineWidth', 2);
    hold on;
    scatter(Mu_chap(pc1, class3), Mu_chap(pc2, class3), 'gx', 'LineWidth', 2);

    
   
    title(['(' num2str(pc1) ',' num2str(pc2) ')']);
    xlabel(['PC' num2str(pc1)]);
    ylabel(['PC' num2str(pc2)]);
    legend(['Classe ' num2str(class1)], ['Classe ' num2str(class2)], ['Classe ' num2str(class3)], ...
       ['μˆ ' num2str(class1) ], ...
       ['μˆ ' num2str(class2) ], ...
       ['μˆ ' num2str(class3) ]);
end



