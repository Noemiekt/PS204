function [phi] = classifieur(x,S,Bx,k,m)
% x = image en entrée
% S = sous-espace vectoriel S engendré par les l eigenfaces les plus énergétiques
% Bx = base d’apprentissage
% k = nombre de plus proche voisin
% m = le nombre d'individu présent dans une classe

x_moy = mean(x); %Moyenne empirique de x

[~,l]=size(S); % l = nb eigenfaces les plus énergétiques (facespace).

w = (x-x_moy).' * S;

[~,Ni]=size(Bx); % Ni = nombres d'images

wbase=zeros(l,Ni); % Stocker les w des images de la bases d'apprentissages.

for i=1:Ni
    xi_moy=mean(Bx(:,i));
    wbase(:,i)=(Bx(:,i)-xi_moy).' * S;
end

distances = zeros(1,Ni);
for i=1:Ni
    distances(i) = sqrt(sum((w'-wbase(:,i)).^2)); % Calcul des distances entre w de l'image et w de la base
end



% Tri des distances et obtention des indices des k plus proches voisins
[~, indices] = sort(distances);

% Sélection des k plus proches voisins
k_plus_proches = indices(1:k);

classes = ceil(k_plus_proches/m);

phi=mode(classes); %classifieur k-NN

end

