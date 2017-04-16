function [y_tr, y_te, coef] = MSECoefficients(nMatTr, nMatrTe, trMat, Index)
% This function finds MSE coefficients for the following inputs:
% - nMatTrain --> normalized training matrix
% - nMatTest --> normalized testing matrix
% - y_tr --> feature used for training
% - y_te --> feature to be tested
% - coef --> MSE coefficients
% [y_tr, y_te, coef] = MSECoefficients(nMatTr,nMatrTe, trMat, Index)

    y_tr = nMatTr(:, Index);  % Feature to be estimated (y_train)
    y_te = nMatrTe(:, Index);   % Feature to be tested (y_test)
    coef = pinv(trMat) * y_tr;
end