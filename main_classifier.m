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

%% Calcul des k

% Indices des sujets
subject_indices = [1, 11, 21, 31, 41, 51]; 

[l_values, k_values, mean_image, l_star] = k_values(subject_indices, data_trn, U, N);


%% Extraction des l_[phi] = classifieur(x,S,Bx,k,m), [phi] = classifieur(x,S,Bx,k,m) star premières colonnes de la matrice U

S = U(:, 1:l_values(l_star-1));
x = data_trn(:,11);
Bx = data_trn;
k = 12;
m = 10; % nobre d'individues presents dans une classe

phi = classifieur(x,S,Bx,k,m);




