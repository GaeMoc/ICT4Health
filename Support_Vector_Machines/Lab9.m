clear all
close all
clc

dataSet = load('arrhythmia.mat');
dataSet = dataSet.arrhythmia;
dataSet(:, ~any(dataSet,1)) = [];
class_id = dataSet(:, end);
ind = find(class_id > 1);
class_id(ind) = -1;
y = dataSet(:, 1:end-1);

N1 = length(find(class_id == 1));
N2 = length(find(class_id == -1));

% ================== LINEAR KERNEL ======
BC = {1, 2, 3, 4, 5, 6};
dim = length(BC);
for ii = 1:dim
    Mdl = fitcsvm(y, class_id, 'BoxConstraint', cell2mat(BC(ii)), ...
        'KernelFunction', 'linear');
    classhat = sign(y * Mdl.Beta + Mdl.Bias);
    detect(ii) = sum((classhat == 1) & (class_id == 1));
    [spec(ii), sens(ii)] = prob2Class(classhat, class_id);
    CVMdl = crossval(Mdl);
    classLoss(ii) = kfoldLoss(CVMdl);    
end

for ii = 1:dim
    Mdl = fitcsvm(y, class_id, 'BoxConstraint', cell2mat(BC(ii)), ...
        'KernelFunction', 'linear');
    classhatG = sign(y * Mdl.Beta + Mdl.Bias);
    detectG(ii) = sum((classhatG == 1) & (class_id == 1));
    [specG(ii), sensG(ii)] = prob2Class(classhatG, class_id);
    CVMdl = crossval(Mdl);
    classLossG(ii) = kfoldLoss(CVMdl);    
end