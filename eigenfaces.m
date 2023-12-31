function [U, Nc, size_cls_trn, Val_non_zero, Vect_non_zero] = eigenfaces(data_trn, lb_trn, P, N)
    
    % Classes contained in the training set
    [~,I]=sort(lb_trn);
    data_trn = data_trn(:,I);
    [cls_trn,bd,~] = unique(lb_trn);
    Nc = length(cls_trn);
    
    % Number of training images in each class
    size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1]; 
    
    % Calcul de XTX
    XTX = data_trn' * data_trn;
    
    % On cherche les vecteurs et valeurs propres
    [Val, Vect] = eig(XTX);
    
    [eigenvalues, sortIdx] = sort(diag(Vect), 'descend');
    Val = Val(:, sortIdx);
    
    % On cherche les valeurs nons nulles
    non_zero_indices = eigenvalues > 0;
    Val_non_zero = Val(:, non_zero_indices);
    Vect_non_zero = Vect(non_zero_indices, non_zero_indices);
    
    % Creation de U
    U = data_trn * Val_non_zero / sqrtm(Vect_non_zero);
    U = normc(U);

end 





