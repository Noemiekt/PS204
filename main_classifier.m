%% KEGL Noémie, SALLMONE Armela & MONY Alexandra
clc;
clear all;
close all;

%% 4) Classification - Classifieur k-NN
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


%% Matrice de confusion

S = U(:, 1:l_values(l_star-1));
Bx = data_trn;
k = 20;
Nc = 10; 
Nc_test = Ni_test/6;

MatConf=zeros(6,6);
 
for c = 1:6
    for ind = 1:Nc_test
        x = data_trn_test(:,(c-1)*Nc_test+ind);
        phi = classifieur(x,S,Bx,k,Nc);
        MatConf(c,phi)=MatConf(c,phi)+1;
    end
end

MatConf=round(MatConf./Nc_test,2);
format shortg;
disp (MatConf);









