function [coeff, L] = PCRCoefficients(nMatTr, nMatTe, ...
    nRows, trMat, perc, index)

    y_tr = nMatTr(:, index);
    y_te = nMatTe(:, index);

    R = (1/nRows) * (trMat') * trMat;   % Covariance matrix
    [U, Lambda] = eig(R);
    Lambdas = diag(Lambda);
    P = sum(Lambdas);
%     Z = x_train * U;    % We map initial features on orthogonal vectors
    somma = 0;
    L = 0;    
    while somma < perc * P
        L = L + 1;
        somma = somma + Lambdas(L);
    end
    LambdaL = Lambda(1:L, 1:L);
%     LambdasL = diag(LambdaL);
    UL = U(:, 1:L);
    aHatPCRL = (1/nRows) * UL * (inv(LambdaL)) * (UL') * (trMat') * y_tr;
    coeff = aHatPCRL;
end