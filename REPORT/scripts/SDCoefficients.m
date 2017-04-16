function [y_tr, y_te, coeff, count] = SDCoefficients(nMatTr, nMatTe, ...
    nF, trMat, t, Index)

    y_tr = nMatTr(:, Index);
    y_te = nMatTe(:, Index);
    
    aHatInitialSD = rand(nF, 1);
    gradientSD = (-2 * (trMat)' * y_tr) + (2 * (trMat)' * ...
                  trMat * aHatInitialSD);
    hessianAHat = 4 * (trMat') * trMat;
    g = ((norm(gradientSD)^2) / (gradientSD' * hessianAHat * ...
               gradientSD));
    aHatFinalSD = aHatInitialSD - (g * gradientSD);    
    ii = 1;
    count = 0;

    while norm(aHatFinalSD - aHatInitialSD) > t
        count(ii) = count(ii) + 1;
        aHatInitialSD = aHatFinalSD;
        gradientSD = (-2 * (trMat') * y_tr) + (2 * (trMat)' *  ...
                      trMat * aHatInitialSD);
        g = ((norm(gradientSD)^2) / (gradientSD' * hessianAHat * ...
                   gradientSD));
        aHatFinalSD = aHatInitialSD - (g * gradientSD);
    end    
    coeff = aHatFinalSD;
end