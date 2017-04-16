function [y_tr, y_te, coeff, count] = GACoefficients(nMatTr, nMatTe, ...
    nF, trMat, g, t, Index)
% - nMatTrain --> normalized training matrix
% - nMatTest --> normalized testing matrix
% - g --> gamma
% - t --> threshold
% - nF --> number of features
% - trMat --> train matrix
% [y_tr, y_te, coeff, count] = GACoefficients(nMatTr, nMatTe, ...
%    nF, trMat, g, t, Index)    

    y_tr = nMatTr(:, Index);
    y_te = nMatTe(:, Index);
    aHatInitialGA = rand(nF, 1);
    gradientGA = (-2 * (trMat)' * y_tr) + (2 * (trMat)' * trMat * ...
        aHatInitialGA);
    aHatFinalGA = aHatInitialGA - (g * gradientGA);    
    ii = 1;
    count = 0;
    % aHatInitial = a(i)
    % aHatFinal = a(i + 1) --> Final coefficients vector
    while norm(aHatFinalGA - aHatInitialGA) > t
        count(ii) = count(ii) + 1;
        aHatInitialGA = aHatFinalGA;
        gradientGA = (-2 * trMat' * y_tr) + (2 * (trMat)' * trMat * ...
            aHatInitialGA);
        aHatFinalGA = aHatInitialGA - (g * gradientGA);
    end    
    coeff = aHatFinalGA;
end