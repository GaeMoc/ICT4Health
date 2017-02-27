function [] = estimPlotPCR(c_F, c_L, index, y_tr, y_tr_N, y_tr_L)

    F = length(c_F);
    L = length(c_L);
    
    v_ = sort(y_tr);
    a_ = v_(1);
    b_ = v_(end);
    l_ = length(y_tr);
    x_ax = linspace(a_, b_, l_);
    y_ = x_ax;
    
    figure, subplot(2,2,1:2)
    plot(c_F), grid on, hold on, plot(c_L, 'o')
    legend(['$\hat{a}$\_N with N = ', num2str(F), ' features'], ...
        ['$\hat{a}$\_L with L = ', num2str(L), ' features'], ...
        'Location', 'northwest')
    title(['PCR: Coefficients for F0 = ', num2str(index)])
    
    subplot(2,2,3)
    plot(y_tr, y_tr_N, 'o'), grid on, hold on,
    plot(x_ax, y_, 'linewidth', 2)
    title(['PCR: Regression for N = ', num2str(F), ...
        ' features and F0 = ', num2str(index)])
    xlabel('y\_train'), ylabel('$\hat{y}$\_train')
    subplot(2,2,4)
    plot(y_tr, y_tr_L, 'o'), grid on, hold on,
    plot(x_ax, y_, 'linewidth', 2)
    title(['PCR: Regression for N = ', num2str(L), ...
        ' features and F0 = ', num2str(index)])
    xlabel('y\_train'), ylabel('$\hat{y}$\_train')    
end