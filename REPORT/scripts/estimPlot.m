function [] = estimPlot(tr, trH, te, teH, featureIndex, coeff, str)
% tr --> vector y_train
% thH --> vector y_train_hat (ESTIMATED)
% te --> vector y_test
% teH --> vector y_test_hat (ESTIMATED)
% featureIndex --> index of the feature
% coeff --> MSE coefficients
% estimPlot(tr, trH, te, teH, featureIndex, coeff)
    
    v_ = sort(tr);
    a = v_(1);
    b = v_(end);
    l = length(tr);
%     x_ax = linspace(a, b, l);
%     y = x_ax;
    
    figure, subplot(2,3,1)
    plot(trH, '--*'), hold on, grid on, plot(tr, '')
    title([str, ' regression: TRAIN F0 = ', num2str(featureIndex)]) 
    legend('$\hat{y}$\_train', 'y\_train'), xlabel('Entries')
    
    subplot(2,3,4)
    plot(teH, '--*'), hold on, grid on, plot(te, '')
    title([str, ' regression: TEST F0 = ', num2str(featureIndex)])
    legend('$\hat{y}$\_test', 'y\_test'), xlabel('Entries')
   
    subplot(2,3,2)
    plot(tr, trH, 'o'), grid on, 
%     plot(x_ax, y, 'linewidth', 2)
    title(['Regression for F0 = ', num2str(featureIndex)])
    xlabel('y\_train'), ylabel('$\hat{y}$\_train')      
    
    subplot(2,3,5)
    plot(coeff), grid on
    title(['Coefficients w for F0 = ', num2str(featureIndex)])   
    
    errTrainMSE = tr - trH;
    varTrainMSE = var(errTrainMSE);
    meanTrainMSE = mean(errTrainMSE);
    subplot(2,3,3)
    hist(errTrainMSE, 50), grid on
    title(['TRAIN prediction error F0 = ', num2str(featureIndex), ...
        '. Var = ', num2str(varTrainMSE), '. Mean = ', num2str(meanTrainMSE)])    

    errTestMSE = te - teH;
    varTestMSE = var(errTestMSE);
    meanTestMSE = mean(errTestMSE);    
    subplot(2,3,6)
    hist(errTestMSE, 50), grid on
    title(['TEST prediction error for F0 = ', num2str(featureIndex), ...
        '. Var = ', num2str(varTestMSE), '. Mean = ', num2str(meanTestMSE)])    
end