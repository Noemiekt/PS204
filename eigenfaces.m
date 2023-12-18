function [U, Nc, size_cls_trn] = eigenfaces(data_trn, lb_trn, P, N)
    
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
    [V, D] = eig(XTX);
    
    [eigenvalues, sortIdx] = sort(diag(D), 'descend');
    V = V(:, sortIdx);
    
    % On cherche les zeros
    non_zero_indices = eigenvalues > 0;
    V_non_zero = V(:, non_zero_indices);
    D_non_zero = D(non_zero_indices, non_zero_indices);
    
    % Creation de U
    U = data_trn * V_non_zero / sqrtm(D_non_zero);
    U = normc(U);

end 





