clear all
close all
clc

% ========================= DATA LOADING ==================================
% =========================================================================
data = load('data2.mat');
chron = data.chron;
N = length(chron(:, 1)); % Number of patients
F = length(chron(1, :));  % Number of features
keylist={'normal','abnormal','present','notpresent','yes','no','good', ...
    'poor','ckd','notckd','?',''};
keymap=[0,1,0,1,0,1,0,1,2,1,NaN,NaN];

for ii = 1:N
    for aa = 1:F
        c = strtrim(chron{ii, aa});
        % check stores the comparison vector between c and keylist
        check = strcmp(c, keylist);
        if sum(check) == 0
            b(ii, aa) = str2double(c);
        else
            % if there is a match between keylist and check, substitute the
            % corresponding value
            index = find(check == 1);   % Indice corrispondente alla keylist
            b(ii, aa) = keymap(index);
        end
    end
end

% ========================= HIERARCHICAL CLUSTERING =======================
% =========================================================================
k = 2;
est = chron(:, end);
kidney = b(:, 1:(end-1));
numEst = b(:, end);

distance = pdist(kidney);
tree1 = linkage(distance);
tree2 = linkage(distance, 'average');
tree3 = linkage(distance, 'centroid');
tree4 = linkage(distance, 'complete');
T1 = cluster(tree1, 'maxclust', k);
T2 = cluster(tree2, 'maxclust', k);
T3 = cluster(tree3, 'maxclust', k);
T4 = cluster(tree4, 'maxclust', k);
p = 0;

error1 = T1 - b(:, end);
det1 = (length(find(error1 == 0)) / N) * 100;
error2 = T2 - b(:, end);
det2 = (length(find(error2 == 0)) / N) * 100;
error3 = T3 - b(:, end);
det3 = (length(find(error3 == 0)) / N) * 100;
error4 = T4 - b(:, end);
det4 = (length(find(error4 == 0)) / N) * 100;

figure, subplot(2,2,1), dendrogram(tree1, p),
title(['single - true detection = ', num2str(det1), '%'])
subplot(2,2,2), dendrogram(tree2, p), title(['average - true detection = ', ...
    num2str(det2), '%'])
subplot(2,2,3), dendrogram(tree3, p), title(['centroid - true detection = ', ...
    num2str(det3), '%'])
subplot(2,2,4), dendrogram(tree4, p), title(['complete - true detection = ', ...
    num2str(det4), '%'])

% figure, plot(error1), grid on
% figure, plot(error2), grid on
% figure, plot(error3), grid on
% figure, plot(error4), grid on
% ======================== HIERARCHICAL CLASSIFICATION ====================
% =========================================================================
tc = fitctree(kidney, est);
view(tc, 'Mode', 'graph');

for aa = 1:N
    if kidney(aa, 15) < 13.05
        if kidney(aa, 16) < 44.5
            classification(aa) = 2;
        else
            classification(aa) = 1;
        end
    else
        if kidney(aa, 3) < 1.0175
            classifcation(aa) = 2;
        else
            if kidney(aa, 4) < 0.5
                classification(aa) = 1;
            else
                classification(aa) = 2;
            end
        end
    end
end

correctProb = classification' - numEst;
correctProb = length(correctProb(correctProb == 0)) / N;