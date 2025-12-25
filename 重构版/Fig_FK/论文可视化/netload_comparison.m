function netload_comparison()
    % 时间轴（24小时）
    t = 0:23;
    % 手动填写优化前后净负荷数据（适合纵坐标20~40MW）
    netload_before = [24 24.2 27 24.6 24 22 22.8 28 31 32 30.1 27.9 25 24.5 24 28 30 32 36 34 36.4 34 32 31];
    netload_after  = [24 24.4 24.4 27 26.9 27.5 27.2 27.3 27.2 27.5 27.7 28 28.1 28.3 28.6 28.9 29.1 29.4 30.9 30 31.7 32.5 31.7 30.8];

    % 绘图
    figure('Position', [180, 180, 500, 350], 'Color', 'white');
    hold on;
    h1 = plot(t, netload_before, '-o', 'Color', [0.85, 0.33, 0.33], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.85, 0.33, 0.33]);
    h2 = plot(t, netload_after,  '-s', 'Color', [0.33, 0.66, 0.33], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.33, 0.66, 0.33]);

    % 坐标轴
    xlabel('时间 (h)', 'FontSize', 12, 'FontName', 'TimesSimSun');
    ylabel('电网负荷 (MW)', 'FontSize', 12, 'FontName', 'TimesSimSun');
    set(gca, 'FontSize', 12, 'FontName', 'TimesSimSun');
    set(gca, 'LineWidth', 1.2);
    set(gca, 'Box', 'off');
    set(gca, 'Color', 'white');
    set(gca, 'XAxisLocation', 'bottom');
    set(gca, 'YAxisLocation', 'left');
    set(gca, 'XColor', 'k', 'YColor', 'k');
    set(gca, 'TickDir', 'in');
    set(gca, 'TickLength', [0.01 0.01]);
    set(gca, 'Position', [0.1, 0.15, 0.85, 0.75]);
    xlim([0, 23]);
    ylim([20, 40]);
    grid off;

    % 图例
    legend([h1, h2], {'优化前', '优化后'}, ...
        'Location', 'northwest', 'FontSize', 12, 'FontName', 'TimesSimSun', 'Box', 'off');

    hold off;

    % 保存图片
    savePath = 'results';
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [8, 6]);
    fileName = 'netload_comparison';
    print(gcf, fullfile(savePath, [fileName, '.png']), '-dpng', '-r600');
    print(gcf, fullfile(savePath, [fileName, '.pdf']), '-dpdf', '-r600');
    print(gcf, fullfile(savePath, [fileName, '.emf']), '-dmeta', '-r600');
    fprintf('净负荷对比图已保存为: %s\n', fileName);
end