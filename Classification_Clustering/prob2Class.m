function [specificity, sensitivity] = prob2Class(estim, trueV)

    truePositive = 0;
    trueNegative = 0;
    falsePositive = 0;
    falseNegative = 0;
    nOfPatients = length(estim(:, 1));

    for ii = 1:nOfPatients
        if estim(ii) == 1 && trueV(ii) == 1
            trueNegative = trueNegative + 1;
        elseif estim(ii) == 2 && trueV(ii) == 2
            truePositive = truePositive + 1;
        elseif estim(ii) == 2 && trueV(ii) == 1
            falsePositive = falsePositive + 1;
        elseif estim(ii) == 1 && trueV(ii) == 2
            falseNegative = falseNegative + 1;
        end    
    end

    sensitivity = truePositive / (truePositive + falseNegative);
%     falseNegativeProb = 1 - sensitivity;
    specificity = trueNegative / (trueNegative + falsePositive);
%     falsePositiveProb = 1 - specificity;    

end