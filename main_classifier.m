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
[P,Ni] = size(data_trn);


% Test set
adr_test = './database/test1/';
fld_test = dir(adr_test);
nb_elt_test = length(fld_test);
% Data matrix containing the training images in its columns 
data_trn_test = []; 
% Vector containing the class of each training image
lb_trn_test = []; 
for i=1:nb_elt_test
    if fld_test(i).isdir == false
        lb_trn_test = [lb_trn_test ; str2num(fld_test(i).name(6:7))];
        img_test = double(imread([adr_test fld_test(i).name]));
        data_trn_test = [data_trn_test img_test(:)];
    end
end
% Size of the test set
[P_test,Ni_test] = size(data_trn_test);



% Calcul de U
[U, m, size_cls_trn] = eigenfaces(data_trn, lb_trn, P, Ni);

% Calcul des k
subject_indices = [1, 11, 21, 31, 41, 51]; 
[l_values, k_values, mean_image] = k_values(subject_indices, data_trn, U, Ni);
l_star = find(k_values >= 0.9, 1, 'first');


%% Extraction des l_[phi] = classifieur(x,S,Bx,k,m), [phi] = classifieur(x,S,Bx,k,m) star premières colonnes de la matrice U

S = U(:, 1:l_values(l_star-1));
Bx = data_trn;
 
% % Avec un x de l'entrainement et k>m
% x = data_trn(:,11);
% k = 12;
% Nc = 10; % nobre d'individues presents dans une classe
% phi_train_kg = classifieur(x,S,Bx,k,Nc);
% 
% % Avec un x de l'entrainement et k<m
% x = data_trn(:,11);
% k = 4;
% Nc = 10; % nobre d'individues presents dans une classe
% phi_train_kp = classifieur(x,S,Bx,k,Nc);

% Avec un x des test et k>m
x = data_trn_test(:,11);
k = 12;
Nc = 10; % nobre d'individues presents dans une classe
phi_train_kg_test = classifieur(x,S,Bx,k,Nc);
% 
% % Avec un x des test et k<m
% x = data_trn_test(:,11);
% k = 4;
% Nc = 10; % nobre d'individues presents dans une classe
% phi_train_kp_test = classifieur(x,S,Bx,k,Nc);

% disp("Avec un x de l'entrainement et k>m:");
% disp(phi_train_kg);
% 
% disp("Avec un x de l'entrainement et k<m:");
% disp(phi_train_kp);

disp("Avec un x du test1 et k>m:");
disp(phi_train_kg_test);

% disp("Avec un x du test1 et k<m:");
% disp(phi_train_kp_test);


