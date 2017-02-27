function [trainNorm, testNorm] = matNorm(updrsNew, tr)
% This function returns the train and test NORMALIZED matrices.
% updrsNew           --> sorted matrix, output of dataLoading function
% tr                 --> number of patients used as train dataset
% trainNorm/testNorm --> output normalized matrixes
% [trainNorm, testNorm] = matNorm(updrsNew, tr)

    trainIndex = find(updrsNew(:, 1) <= tr);
    data_train = updrsNew(1:length(trainIndex), :);
    data_test = updrsNew(length(trainIndex)+1:end, :);
    m_data_train = mean(data_train, 1);
    v_data_train = var(data_train, 1);
    trainDim = length(data_train(:, 1));
    testDim = length(data_test(:, 1));
    onesMatrixTrain = ones(trainDim, 1);
    onesMatrixTest = ones(testDim, 1);
    meanMatrixTrain = onesMatrixTrain * m_data_train;
    meanMatrixTest = onesMatrixTest * m_data_train;
    varMatrixTrain = onesMatrixTrain * v_data_train;
    varMatrixTest = onesMatrixTest * v_data_train;
    trainNorm = (data_train - meanMatrixTrain) ./ sqrt(varMatrixTrain);
    testNorm = (data_test - meanMatrixTest) ./ sqrt(varMatrixTest);

end