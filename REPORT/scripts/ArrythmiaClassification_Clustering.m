clear all
close all
clc

% LEGEND:
% arrhythmia        --> original data
% classIdOriginal   --> Original column with patients classes
% y                 --> Feature matrix WITHOUT classes
% yNorm             --> Feature matrix WITHOUT classes
% pi16              --> Vector containing the prob related to each class
% xxxNClassesMDC    --> Related to MINIMUM DISTANCE CRITERION N classes
% xxxNClassesBay    --> Related to BAYES CRITERION N classes
% xxxMeans          --> Matrix ORDERED with the means of the features per 
                        % each class
set(groot,'DefaultLegendInterpreter','latex')
set(groot,'DefaultTextInterpreter','latex')
% ========================== MATRIX LOADING ===============================
load('arrhythmia.mat')

% ============================= DATA PREPARING ============================
arr = arrhythmia;   % Original dataset
nOfPatients = length(arr(:, 1));
maxClassId = max(arr(:, end));
classIdOriginal = arr(:, end);
% Substitution of the last column with 1-2 classes
results = arr(:, end);  
results(results > 1) = 2;
arr(:, end) = results;

% This command deletes all colomns that don't carry information (empty)
arr(:, ~any(arr,1)) = [];
class_id2Classes = arr(:, end);     % Class ID (just 2 classes)
y = arr(:, 1:(end - 1));    % Features matrix without class ID colomn
F = length(y(1, :));
meanY = mean(y, 1);         % Vector containing  the MEAN per feature
varY = var(y, 1);           % Vector containing the VARIANCE per feature

% NORMALIZATION of the features matrix
meanYMatrix = ones(nOfPatients, 1) * meanY;
varYMatrix = ones(nOfPatients, 1) * varY;
yNorm = (y - meanYMatrix) ./ sqrt(varYMatrix);

indexClass1 = find(class_id2Classes == 1);   % classID 1 indices
indexClass2 = find(class_id2Classes == 2);   % classID 2 indices
y1 = yNorm(indexClass1, :);    % Class 1 matrix, WITHOUT CLASSID
y2 = yNorm(indexClass2, :);    % Class 2 matrix, WITHOUT CLASSID
x1 = mean(y1, 1);   % Row vector of the mean per each feature for class 1
x2 = mean(y2, 1);   % Row vector of the mean per each feature for class 2

% Matrix with x1 and x2: first row contains the mean of the features of the
% FIRST class, second row contains the mean of the features of the SECOND
% class:
xMeans = [x1; x2];  

% ============================== 2 CLASSES ================================
% ===================== MINIMUM DISTANCE CRITERION ========================
eny = diag(yNorm * yNorm');
enx = diag(xMeans * xMeans');   % Region centroids
dotProd = yNorm * xMeans';
[U, V] = meshgrid(enx, eny);

% Matrix containing distance of EACH patient (row) from each class
distance2Classes = U + V - 2*dotProd;   
[val, ind] = min(distance2Classes, [], 2);  % <-- MIN DISTANCE EVALUATION
est_class_id_2ClassesMDC = ind;

% --------------------- Sensitivity & Specificity -------------------------
[specificityMDC, sensitivityMDC] = prob2Class(est_class_id_2ClassesMDC, ...
    class_id2Classes);

figure, subplot(2,1,1)
plot(est_class_id_2ClassesMDC, 'o'), hold on, grid on
plot(class_id2Classes, '*'), title(['MDC plot 2 classes: Sensitivity = ', ...
    num2str(sensitivityMDC), ' Specificity = ', num2str(specificityMDC)])
legend('Class ID estimation', 'Class ID true')

% ============================= 2 CLASSES =================================
% ======================== BAYESIAN CRITERION =============================
% First of all I evaluate the prior probability for each class.
pi1 = length(indexClass1) / nOfPatients;    % Prob that hyp 1 is correct
pi2 = length(indexClass2) / nOfPatients;    % Prob that hyp 2 is correct
R = (1/nOfPatients) * (yNorm') * (yNorm);   % Covariance matrix
[U2Classes, Lambda2Classes] = eig(R);  % U2Classes columns are EIGENVECTORS    
Lambdas2Classes = diag(Lambda2Classes);
sommaLambdas2Classes = sum(Lambdas2Classes);
P2Classes = 0.99;   % We take the 99% of the total number of eigenvalues
somm2Classes = 0;
ii = 0;

while somm2Classes < P2Classes * sommaLambdas2Classes
    ii = ii + 1;
    somm2Classes = somm2Classes + Lambdas2Classes(ii); 
end

UL2Classes = U2Classes(:, 1:ii);    % I just take the 99% of eigenvectors
z2Classes = yNorm * UL2Classes; 

% z2Classes is the projection of original features in an new orthoGONAL
% space. We need to re-normalize again to get orthoNORMAL vectors of
% features.
% ------------------------ z normalization --------------------------------
zMean2Classes = ones(nOfPatients, 1) * mean(z2Classes, 1);
zVar2Classes = ones(nOfPatients, 1) * var(z2Classes, 1);
% zNorm2Classes is now the orthonormal set of features:
zNorm2Classes = (z2Classes - zMean2Classes) ./ sqrt(zVar2Classes);

% I now perform again the minimum distance criterion with the new set of
% features:
w1 = mean(zNorm2Classes(indexClass1, :), 1);
w2 = mean(zNorm2Classes(indexClass2, :), 1);
wMeans = [w1; w2];
eny = diag(zNorm2Classes * zNorm2Classes');
enx = diag(wMeans * wMeans');
dotProd = zNorm2Classes * wMeans';
[U, V] = meshgrid(enx, eny);
% This matrix contains the distance of every patient (row) from the
% relative class (column): 
distance2ClassesBay = U + V - 2*dotProd;  
% est_class_id_2ClassesBay will be a vector containing the estimation of
% the minimum distance of each patient (row) from class 1 (column 1) and
% class 2 (column 2):

pi1Vector = ones(nOfPatients, 1) * (2 * log(pi1));
pi2Vector = ones(nOfPatients, 1) * (2 * log(pi2));
class1 = distance2ClassesBay(:, 1) - pi1Vector;
class2 = distance2ClassesBay(:, 2) - pi2Vector;
distance2ClassesBay = [class1 class2];
[value, est_class_id_2ClassesBay] = min(distance2ClassesBay, [], 2);
% [value, est_class_id_2ClassesBay] = min(distance2ClassesBay, [], 2);
% est_class_id_2ClassesBay = est_class_id_2ClassesBay';

% --------------------- Sensitivity & Specificity --------------------
[specificity2ClassesBay, sensitivity2ClassesBay] = prob2Class(est_class_id_2ClassesBay, ...
    class_id2Classes);

subplot(2,1,2), plot(est_class_id_2ClassesBay, 'o'), hold on, grid on
plot(class_id2Classes, '*'), title(['Bayes criterion plot 2 classes: Sensitivity = ', ...
    num2str(sensitivity2ClassesBay), ' Specificity = ', ...
    num2str(specificity2ClassesBay)])
legend('Class ID estimation', 'Class ID true')

% ========================== 16 CLASSES ===================================
% ===================== MINIMUM DISTANCE CRITERION ========================
% Prior probability for each class is stored in pi16Classes:
for aa = 1:maxClassId
    indexes = find(classIdOriginal == aa);
    pi16Classes(aa) = length(indexes) / nOfPatients;
    xMean16Classes(aa, :) = mean(yNorm(indexes, :), 1);
end

eny16Classes = diag(yNorm * yNorm');
enx16Classes = diag(xMean16Classes * xMean16Classes');
dotProd16Classes = yNorm * xMean16Classes';
[U16Classes, V16Classes] = meshgrid(enx16Classes, eny16Classes);
% distance16Classes stores the distance of each patient from each class
distance16Classes = U16Classes + V16Classes - 2*dotProd16Classes;

% Performance evaluation: I distinguish the true detection from the false
% detection:
trueDetection16ClassesMDC = 0;
falseDetection16ClassesMDC = 0;
[minVal, est_class_id_16ClassesMDC] = min(distance16Classes, [], 2);
diff16Classes = est_class_id_16ClassesMDC - classIdOriginal;
trueDetection16ClassesMDC = length(find(diff16Classes == 0));
falseDetection16ClassesMDC = length(find(diff16Classes ~= 0));

percTrueDetection16ClassesMDC = trueDetection16ClassesMDC / nOfPatients;
percFalseDetection16ClassesMDC = falseDetection16ClassesMDC / nOfPatients;

figure, plot(est_class_id_16ClassesMDC, 'o'), hold on, grid on
plot(classIdOriginal, '*'), title(['MDC plot 16 classes: True detection = ', ...
    num2str(percTrueDetection16ClassesMDC), ' False detection = ', ...
    num2str(percFalseDetection16ClassesMDC)])
legend('Class ID estimation', 'Class ID true')

% ========================== 16 CLASSES ===================================
% ========================= BAYESIAN CRITERION ============================
% Covariance matrix for the original dataset:
R16Classes = (1/nOfPatients) * (yNorm') * (yNorm);   
[U16Classes, Lambda16Classes] = eig(R16Classes);               
Lambdas16Classes = diag(Lambda16Classes);
sommaLambdas16Classes = sum(Lambdas16Classes);
P16Classes = 0.999;
somm16Classes = 0;
ii16Classes = 0;

while somm16Classes < P16Classes * sommaLambdas16Classes
    ii16Classes = ii16Classes + 1;
    somm16Classes = somm16Classes + Lambdas16Classes(ii16Classes); 
end

UL16Classes = U16Classes(:, 1:ii16Classes);
z16Classes = yNorm * UL16Classes;
zMean16Classes = ones(nOfPatients, 1) * mean(z16Classes, 1);
zVar16Classes = ones(nOfPatients, 1) * var(z16Classes, 1);
zNorm16Classes = (z16Classes - zMean16Classes) ./ ... 
    sqrt(zVar16Classes);

% Matrix sorting:
for aa = 1:maxClassId
    indici16Classes = find(classIdOriginal == aa);
    wMean16ClassesBay(aa, :) = mean(zNorm16Classes(indici16Classes, :), 1);
end

eny16ClassesBay = diag(zNorm16Classes * zNorm16Classes');
enx16ClassesBay = diag(wMean16ClassesBay * wMean16ClassesBay');
dotProd16ClassesBay = zNorm16Classes * wMean16ClassesBay';
[U16ClassesBay, V16ClassesBay] = meshgrid(enx16ClassesBay, eny16ClassesBay);
distanceBay16ClassesBay = U16ClassesBay + V16ClassesBay - 2*dotProd16ClassesBay;
pi16Matrix = ones(nOfPatients, 1) * (2 * log(pi16Classes));
bayesDist = distanceBay16ClassesBay - pi16Matrix;
[minimum, est_class_id_16ClassesBay] = min(bayesDist, [], 2);

probC = 0;

for gg = 1:16
    estimatedPatients = 0;
    for oo = 1:nOfPatients
        if est_class_id_16ClassesBay(oo) == gg && classIdOriginal(oo) == gg
            estimatedPatients = estimatedPatients + 1;
        end 
    end
    truePatients = length(find(classIdOriginal == gg));
    if truePatients == 0
    else
        % Probability of right decision:
        probC = probC + (estimatedPatients / truePatients) * pi16Classes(gg);
    end
end
% ------------------------------ FIGURES ----------------------------------
figure, subplot(2,1,1)
plot(est_class_id_16ClassesMDC, '*'), hold on, grid on
    plot(classIdOriginal, 'o'), legend('ClassID estimation', 'ClassID true')
title(['Class detection MDC plot: true detection = ', ...
    num2str(percTrueDetection16ClassesMDC * 100), ' %'])

subplot(2,1,2)
plot(est_class_id_16ClassesBay, '*'), hold on, grid on
    plot(classIdOriginal, 'o'), legend('ClassID estimation', 'ClassID true')
title(['Class detection Bayesian criterion plot: true detection = ', ...
    num2str(probC * 100), ' %'])

confMat16ClassesMDC = confusionmat(classIdOriginal, est_class_id_16ClassesMDC);
confMat16ClassesBay = confusionmat(classIdOriginal, est_class_id_16ClassesBay);

figure, subplot(1,2,1),
mesh(confMat16ClassesMDC), title('Confusion matrix for MDC with 16 classes')
xlabel('ClassID original'), ylabel('ClassID estimation')
subplot(1,2,2), mesh(confMat16ClassesBay)
title(['Confusion matrix for Bayes criterion with 16 classes with P = ',...
    num2str(P16Classes)])
xlabel('ClassID original'), ylabel('ClassID estimation')

% ================================ HARD K-MEANS ===========================
% ================================ 2 CLUSTERS =============================
% Starting from xMeans already evaluated above.
% LO SCOPO E' QUELLO DI MINIMIZZARE LA VARIANZA INTRA-CLUSTER PER OGNI
% CLUSTER: OGNI CLUSTER VIENE IDENTIFICATO DA UN CENTROIDE
% L'initial step dovrebbe essere una scelta casuale di K clusters. Quindi
% inizializzare un vettore di nOfPatients righe per K colonne dove ogni
% elemento contiene la distanza!!!
yNorm;       % <-- Normalized data
k = 2;        
x_k = xMeans;        % <-- INITIAL GUESS: step 1
sigma_k = ones(1, k);
pi_k = [1/k 1/k];
count = 0;

while count < 10
    for ii = 1:k
        matrix = yNorm - (ones(nOfPatients, 1) * x_k(ii, :));
        norma = sqrt(sum(abs(matrix).^2,2));    
        MAP_values(:, ii) = pi_k(ii) .* exp(-(norma) ./ (2*sigma_k(ii))) / ...
            (2*pi*sigma_k(ii))^(F/2);
    end
    % Assignment step------------------------------------------------------
    [p_, assignm_step] = max(MAP_values, [], 2);
    
    % Update step ---------------------------------------------------------
    for ii = 1:k
        new_indexes = find(assignm_step == ii);
        N(ii) = length(new_indexes);
        pi_k(ii) = N(ii) / nOfPatients; % prior probabilities UPDATED
        w = yNorm(new_indexes, :);
        if ii == 1
            w_1 = yNorm(new_indexes, :);
        else
            w_2 = yNorm(new_indexes, :);
        end
        x_k(ii, :) = mean(w, 1);    % x_k UPDATED
        matrix2 = w - (ones(N(ii), 1) * x_k(ii, :));
        norma2 = sqrt(sum(abs(matrix2).^2,2));
        norma2 = sum(norma2);
        sigma_k(ii) = norma2 / (F * (N(ii) - 1));    % variance UPDATED
%         if count == 5
%             err1(ii, :) = abs(xMeans(ii, :) - x_k(ii, :));
%         end    
    end
    count = count + 1;
end

[specificity_kM, sensitivity_kM] = prob2Class(assignm_step, ...
    class_id2Classes);
figure, plot(assignm_step, 'o'), hold on, grid on
plot(class_id2Classes, '*'), title(['k-means 2 classes: Sensitivity = ', ...
    num2str(sensitivity_kM), ' Specificity = ', num2str(specificity_kM)])
legend('Class ID estimation', 'Class ID true') 

figure, subplot(2,1,1)
plot(xMeans(1, :)), hold on, grid on, plot(x_k(1, :))
title('Difference between initial and final distances - Cluster 1')
legend('Initial distances', 'Final distances')
subplot(2,1,2)
plot(xMeans(2, :)), hold on, grid on, plot(x_k(2, :))
title('Difference between initial and final distances - Cluster 2')
legend('Initial distances', 'Final distances')

figure
% This graph evaluates the consistance of clusters: it tells how well each
% object lies within each cluster.
[silh2, h] = silhouette(yNorm, assignm_step, 'cityblock');
grid on
savg = grpstats(silh2,assignm_step);
title('Silhouette plot - From classification results')
% 
% [coeffy1, scorey1] = pca(y1);
% [coeffy1w, scorey1w] = pca(w_1);
% [coeffy2, scorey2] = pca(y2);
% [coeffy2w, scorey2w] = pca(w_2);
% numDimen = 2;
% scorey1red = scorey1(:, 1:numDimen);
% a = scorey1red;
% scorey1redw = scorey1w(:, 1:numDimen);
% media1Init = mean(scorey1red, 1);
% media1Final = mean(scorey1redw, 1);
% scorey2red = scorey2(:, 1:numDimen);
% b = scorey2red;
% scorey2redw = scorey2w(:, 1:numDimen);
% media2Init = mean(scorey2red, 1);
% media2Final = mean(scorey2redw, 1);
% 
% figure, plot(scorey1red(:, 1), scorey1red(:, 2), 'rx', ...
%     scorey2red(:, 1), scorey2red(:, 2), 'bo'), grid on
% title('Patient clustering'), xlabel('x'), ylabel('y')
% 
% figure, plot(media1Init(:, 1), media1Init(:, 2), 'kx'), ...
%     hold on, plot(media2Init(:, 1), media2Init(:, 2), 'ko')
% hold on, plot(media1Final(:, 1), media1Final(:, 2), 'rx'), ...
%     hold on, plot(media2Final(:, 1), media2Final(:, 2), 'ro'), grid on
% legend('Initial cluster 1', 'Initial cluster 2', 'Final cluster 1', ...
%     'Final cluster 2', 'Location', 'Best'), xlabel('x'), ylabel('y')
% title('Centroid positions - From classification results')

% Clustering starting from random distances vectors -----------------------
yNorm;       % <-- Normalized data
k = 2;        
x_k = rand(k, F);        % <-- INITIAL GUESS: step 1
sigma_k = ones(1, k);
pi_k = [1/k 1/k];
count = 0;

while count < 100
    for ii = 1:k
        matrix = yNorm - (ones(nOfPatients, 1) * x_k(ii, :));
        norma = sqrt(sum(abs(matrix).^2,2));    % Norma della situazione
        MAP_values(:, ii) = pi_k(ii) .* exp(-(norma) ./ (2*sigma_k(ii))) / ...
            (2*pi*sigma_k(ii))^(F/2);
    end
    % Assignment step------------------------------------------------------
    % MAX ITERA SULLE COLONNE QUINDI OTTENIAMO IL MASSIMO PER OGNI RIGA
    [p_, assignm_step] = max(MAP_values, [], 2);
    
    % Update step ---------------------------------------------------------
    for ii = 1:k
        new_indexes = find(assignm_step == ii);
        N(ii) = length(new_indexes);
        pi_k(ii) = N(ii) / nOfPatients; % prior probabilities UPDATED
        w = yNorm(new_indexes, :);
        if ii == 1
            w_1 = yNorm(new_indexes, :);
        else
            w_2 = yNorm(new_indexes, :);
        end        
        x_k(ii, :) = mean(w, 1);    % x_k UPDATED
        matrix2 = w - (ones(N(ii), 1) * x_k(ii, :));
        norma2 = sqrt(sum(abs(matrix2).^2,2));
        norma2 = sum(norma2);
        sigma_k(ii) = norma2 / (F * (N(ii) - 1));    % variance UPDATED   
    end
    count = count + 1;
        
end

[specificity_kM, sensitivity_kM] = prob2Class(assignm_step, ...
    class_id2Classes);
figure, plot(assignm_step, 'o'), hold on, grid on
plot(class_id2Classes, '*'), title(['k-means 2 clusters: Sensitivity = ', ...
    num2str(sensitivity_kM), ' Specificity = ', num2str(specificity_kM)])
legend('Class ID estimation', 'Class ID true') 

figure, subplot(2,1,1)
plot(xMeans(1, :)), hold on, grid on, plot(x_k(1, :))
title('Difference between initial and final distances (random) - Cluster 1')
legend('Initial distances', 'Final distances')
subplot(2,1,2)
plot(xMeans(2, :)), hold on, grid on, plot(x_k(2, :))
title('Difference between initial and final distances (random)- Cluster 2')
legend('Initial distances', 'Final distances')

figure
[silh2, h] = silhouette(yNorm, assignm_step, 'cityblock');
grid on
savg = grpstats(silh2,assignm_step);
title('Silhouette plot - From random initial distances')

% [coeffy1, scorey1] = pca(y1);
% [coeffy1w, scorey1w] = pca(w_1);
% [coeffy2, scorey2] = pca(y2);
% [coeffy2w, scorey2w] = pca(w_2);
% numDimen = 2;
% scorey1red = scorey1(:, 1:numDimen);
% scorey1redw = scorey1w(:, 1:numDimen);
% media1Init = mean(scorey1red, 1);
% media1Final = mean(scorey1redw, 1);
% scorey2red = scorey2(:, 1:numDimen);
% scorey2redw = scorey2w(:, 1:numDimen);
% media2Init = mean(scorey2red, 1);
% media2Final = mean(scorey2redw, 1);
% 
% figure, plot(scorey1red(:, 1), scorey1red(:, 2), 'rx', ...
%     scorey2red(:, 1), scorey2red(:, 2), 'bo'), grid on, hold on,
% plot(a(:, 1), a(:, 2), 'gx', b(:, 1), b(:, 2), 'yo')
% 
% title('Patient clustering'), xlabel('x'), ylabel('y')
% figure, plot(media1Init(:, 1), media1Init(:, 2), 'kx'), ...
%     hold on, plot(media2Init(:, 1), media2Init(:, 2), 'ko')
% hold on, plot(media1Final(:, 1), media1Final(:, 2), 'rx'), ...
%     hold on, plot(media2Final(:, 1), media2Final(:, 2), 'ro'), grid on
% legend('Initial cluster 1', 'Initial cluster 2', 'Final cluster 1', ...
%     'Final cluster 2', 'Location', 'Best'), xlabel('x'), ylabel('y')
% title('Centroid positions - From random initial distances')

% ================================ HARD K-MEANS ===========================
% ================================ 4 CLUSTERS =============================
yNorm;      % <-- Normalized data
k = 4;        
x_k1 = rand(k, F);        % <-- INITIAL GUESS: step 1
x_k2 = rand(k, F);
sigma_k1 = ones(1, k);
sigma_k2 = ones(1, k);
pi_k1 = [1/k 1/k 1/k 1/k];
pi_k2 = [1/k 1/k 1/k 1/k];
count = 0;

while count < 10
    for ii = 1:k
        matrix1 = yNorm - (ones(nOfPatients, 1) * x_k1(ii, :));
        norma1 = sqrt(sum(abs(matrix1).^2,2));   
        MAP_values1(:, ii) = pi_k1(ii) .* exp(-(norma1) ./ (2*sigma_k1(ii))) / ...
            (2*pi*sigma_k1(ii))^(F/2);
        
        matrix2 = yNorm - (ones(nOfPatients, 1) * x_k2(ii, :));
        norma2 = sqrt(sum(abs(matrix2).^2,2));   
        MAP_values2(:, ii) = pi_k2(ii) .* exp(-(norma2) ./ (2*sigma_k2(ii))) / ...
            (2*pi*sigma_k2(ii))^(F/2);        
    end
    % Assignment step------------------------------------------------------
    % MAX ITERA SULLE COLONNE QUINDI OTTENIAMO IL MASSIMO PER OGNI RIGA
    [p_, assignm_step1] = max(MAP_values1, [], 2);
    [p_, assignm_step2] = max(MAP_values2, [], 2);
    
    % Update step ---------------------------------------------------------
    for ii = 1:k
        new_indexes1 = find(assignm_step1 == ii);
        N1(ii) = length(new_indexes1);
        pi_k1(ii) = N1(ii) / nOfPatients; % prior probabilities UPDATED
        w1 = yNorm(new_indexes1, :);
        x_k1(ii, :) = mean(w1, 1);    % x_k UPDATED
        matrix12 = w1 - (ones(N1(ii), 1) * x_k1(ii, :));
        norma12 = sqrt(sum(abs(matrix12).^2,2));
        norma12 = sum(norma12);
        sigma_k1(ii) = norma12 / (F * (N1(ii) - 1));    % variance UPDATED 
        
        new_indexes2 = find(assignm_step2 == ii);
        N2(ii) = length(new_indexes2);
        pi_k2(ii) = N2(ii) / nOfPatients; % prior probabilities UPDATED
        w2 = yNorm(new_indexes2, :);
        x_k2(ii, :) = mean(w2, 1);    % x_k UPDATED
        matrix22 = w2 - (ones(N2(ii), 1) * x_k2(ii, :));
        norma22 = sqrt(sum(abs(matrix22).^2,2));
        norma22 = sum(norma22);
        sigma_k2(ii) = norma22 / (F * (N2(ii) - 1));    % variance UPDATED        
    end
    count = count + 1;
end

figure, subplot(1,2,1)
[silh2, h] = silhouette(yNorm, assignm_step1, 'cityblock');
grid on
savg = grpstats(silh2,assignm_step1);
title('Silhouette plot - From random initial distances')

subplot(1,2,2)
[silh2, h] = silhouette(yNorm, assignm_step2, 'cityblock');
grid on
savg = grpstats(silh2,assignm_step2);
title('Silhouette plot - From random initial distances')