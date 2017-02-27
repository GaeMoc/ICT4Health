close all
clear all
clc

load('updrs.mat');
updrs = parkinsonsupdrs;
nOfDays = 180;
nOfPatients = 42;
trainingPatients = 36;
set(groot,'DefaultLegendInterpreter','latex')
set(groot,'DefaultTextInterpreter','latex')

% =================== DATA LOADING & NORMALIZATION ========================
% =========================================================================

% dataLoading function is devoted to "data cleaning" procedure: from the
% imported raw data (updrs) it deletes empty features, averages daily
% measurements per patient and sorts days per patient.
updrsNew = dataLoading(updrs, nOfPatients, nOfDays);

% The obtained matrix must be normalized through "matNorm" function. The
% output is train matrix and test matrix, normalized.
[data_train_norm, data_test_norm] = matNorm(updrsNew, trainingPatients);

consideredFeatures = [2:4 8:22];
trainMatrix = data_train_norm(:, consideredFeatures);
testMatrix = data_test_norm(:, consideredFeatures);
x_train = trainMatrix;
x_test = testMatrix;
% Residual features are: (2,3,4,8,9):end

F0 = [5 7]; % Vector containing analysed features.

tic
% ========================= MSE ESTIMATION ===========================
% ====================================================================
for ii = 1:length(F0)
    % MSECoefficients function returns MSE coefficients and the test and
    % train vectors.
    [y_train, y_test, aHatMSE] = MSECoefficients(data_train_norm, ...
        data_test_norm , x_train, F0(ii));

    y_train_hat = x_train * aHatMSE;       % Trying TRAINING data. 
    y_test_hat = x_test * aHatMSE;         % Trying TESTING data.
    
    estimPlot(y_train, y_train_hat, y_test, y_test_hat, F0(ii), aHatMSE, 'MSE');   
end
timeElapsedMSE = toc

% ========================= GRADIENT ALGORITHM =======================
% ====================================================================
rng('default');
M = length(aHatMSE);       % Number of features
threshold = 10^-6;      % Treshold for the stopping condition
gamma = 10^-5;          % Speed of convergence
% countGA vector contains the number of iterations per each F0:
countGA = zeros(1, length(F0));     

tic 
for ii = 1:length(F0)
    
    [y_train, y_test, aHatGA, countGA(ii)] = GACoefficients(data_train_norm, ...
        data_test_norm, M, x_train, gamma, threshold, F0(ii));

    % In aHatFinal there is the final set of coefficients a(i+1):
    y_train_hat = x_train * aHatGA;
    y_test_hat = x_test * aHatGA;
    
    estimPlot(y_train, y_train_hat, y_test, y_test_hat, F0(ii), aHatGA, 'GA'); 
end
timeElapsedGA = toc

% ========================= STEPEST DESCENT ==========================
% ====================================================================
rng('Default');
% Vector containing the number of iterations per each F0
countSD = zeros(1, length(F0));     

tic
for ii = 1:length(F0)
    [y_train, y_test, aHatSD, countSD(ii)] = SDCoefficients(data_train_norm, ...
        data_test_norm, M, x_train, threshold, F0(ii));
    
    y_train_hat = x_train * aHatSD;
    y_test_hat = x_test * aHatSD;    
    
    estimPlot(y_train, y_train_hat, y_test, y_test_hat, F0(ii), aHatSD, 'SD');
end
timeElapsedSD = toc

% =============================== PCR ================================
% ====================================================================
F = M;  % Numero di features originali
N = length(x_train(:, 1));  
tic 
for ii = 1:length(F0)
    
    [y_train, y_test, aHatPCR] = MSECoefficients(data_train_norm, ...
        data_test_norm , x_train, F0(ii));       
    
    percentage = 0.9;
    % L is the new number of features considered
    [aHatPCRL, L] = PCRCoefficients(data_train_norm, data_test_norm, N, ...
        x_train, percentage, F0(ii));  
        
    % Result computing for both N and L features considered:
    y_train_hat_Nfeature = x_train * aHatPCR;
    y_test_hat_Nfeature = x_test * aHatPCR;
    y_train_hat_Lfeature = x_train * aHatPCRL;
    y_test_hat_Lfeature = x_test * aHatPCRL;
    
    % Task 1
    figure, subplot(2,2,1)
    plot(y_train_hat_Nfeature, '--*'), hold on, grid on, plot(y_train)
    legend(['$\hat{y}$\_train\_N = ', num2str(F), ' features'], ...
        'y\_train', 'Location', 'northwest')
    title(['PCR: N = ', num2str(F), '. TRAIN F0 = ', ...
        num2str(F0(ii))])
    subplot(2,2,2)
    plot(y_train_hat_Lfeature, '--*'), hold on, grid on, plot(y_train)
    legend(['$\hat{y}$\_train\_L = ', num2str(L), ' features'], ...
        'y\_train', 'Location', 'northwest')
    title(['PCR: N = ', num2str(L), '. TRAIN F0 = ', ...
        num2str(F0(ii))])
    
    % Task 2
    subplot(2,2,3)
    plot(y_test_hat_Nfeature, '--*'), hold on, grid on, plot(y_test)
    legend(['$\hat{y}$\_test\_N = ', num2str(F), ' features'], 'y\_test')
    title(['PCR: N = ', num2str(F), '. TEST F0 = ', ...
        num2str(F0(ii))])
    subplot(2,2,4)
    plot(y_test_hat_Lfeature, '--*'), hold on, grid on, plot(y_test)
    legend(['$\hat{y}$\_test\_L = ', num2str(L), ' features'], 'y\_test')
    title(['PCR: N = ', num2str(L), '. TEST F0 = ', ...
        num2str(F0(ii))])
    
    % Task 3
    errTrainPCRN = y_train - y_train_hat_Nfeature;
    meanTrainPCRN = mean(errTrainPCRN);
    varTrainPCRN = var(errTrainPCRN);
    figure, subplot(2,2,1)
    hist(errTrainPCRN, 50), grid on
    title(['TRAIN for N = ', num2str(F), ...
        '. F0 = ', num2str(F0(ii)), '. var = ', num2str(varTrainPCRN)])
    subplot(2,2,2)
    errTrainPCRL = y_train - y_train_hat_Lfeature;
    meanTrainPCRL = mean(errTrainPCRL);
    varTrainPCRL = var(errTrainPCRL);    
    hist(errTrainPCRL, 50), grid on
    title(['TRAIN for N = ', num2str(L), ...
        '. F0 = ', num2str(F0(ii)), '. var = ', num2str(varTrainPCRL)])
    
    % Task 4
    errTestPCRN = y_test - y_test_hat_Nfeature;
    meanTestPCRN = mean(errTestPCRN);
    varTestPCRN = var(errTestPCRN);     
    subplot(2,2,3)
    hist(errTestPCRN, 50), grid on
    title(['TEST for N =  ', num2str(F), ...
        '. F0 = ', num2str(F0(ii)), '. var = ', num2str(varTestPCRN)])
    subplot(2,2,4)
    errTestPCRL = y_test - y_test_hat_Lfeature;
    meanTestPCRL = mean(errTestPCRL);
    varTestPCRL = var(errTestPCRL);       
    hist(y_test - y_test_hat_Lfeature, 50), grid on
    title(['TEST for N = ', num2str(L), ...
        '. F0 = ', num2str(F0(ii)), '. var = ', num2str(varTestPCRL)])
    
    % Task 5
    estimPlotPCR(aHatPCR, aHatPCRL, F0(ii), y_train, ...
        y_train_hat_Nfeature, y_train_hat_Lfeature)
end
timeElapsedPCR = toc