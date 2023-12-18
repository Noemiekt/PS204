function [phi] = classifieurgaussien(x,Bx,m,l,Ni,Nc, S)
%% Données
% x = image en entrée
% Bx = base d’apprentissage
% m = nombre de classse
% Ni = nombres d'individus
% Nc = nombres d'individus par classe
% l = nb eigenfaces les plus énergétiques (facespace).

%% Base d'entrainement
%% Composantes principales

wbase=zeros(l,Ni); % Stocker les w des images de la bases d'apprentissages.

for i=1:Ni
    xi_moy=mean(Bx(:,i));
    wbase(:,i)=(Bx(:,i)-xi_moy).' * S;
end

%% Moyenne intra classe :

Mu_chap=zeros(l,m);

for i = 1:m
    % Index des colonnes du groupe actuel
    iC = (i - 1) * Nc + 1 : i * Nc;

    % Additionner les colonnes du groupe actuel
    Mu_chap(:, i) = sum(wbase(:, iC), 2);
end

%% Matrice de covariance empirique

Sigma_chap=zeros(l,l);

for j=1:m
    for i=1:Nc
        tmp=wbase(:,(j-1)*Nc+i)-Mu_chap(:,j);
        Sigma_chap=Sigma_chap+tmp*tmp';
    end
end

Sigma_chap = Sigma_chap/Ni;

%% Traitement
%% Composantes principales
x_moy = mean(x); %Moyenne empirique de x
w = (x-x_moy).' * S;

%% Maximum de vraisemblances

phi_tab=zeros(1,m);
for j=1:m
    tmp=w'-Mu_chap(:, j);
    phi_tab(j)=sqrt(sum((Sigma_chap^(-1/2)*tmp).^2));
end

disp(phi_tab);
phi=floor(min(phi_tab));


end

