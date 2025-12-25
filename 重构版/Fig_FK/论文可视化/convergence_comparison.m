function convergence_comparison()
    % 迭代次数
    iter = 0:500;
    % 起点相同，收敛速度和最小适应度不同（示例数据，可自行替换）
    y0 = 4200;
    gwo  = y0 * exp(-0.312*iter) + 900;   % GWO
    pso  = y0 * exp(-0.078*iter) + 280;   % PSO
    issa = y0 * exp(-0.125*iter) + 150;   % ISSA
    ssa  = y0 * exp(-0.092*iter) + 250;   % SSA
    cso  = y0 * exp(-0.050*iter) + 380;   % CSO

    % 绘图
    figure('Position', [180, 180, 500, 350], 'Color', 'white');
    hold on;
    h1 = plot(iter, gwo, 'Color', [0 0.5 0], 'LineWidth', 2);      % 深绿
    h2 = plot(iter, pso, 'Color', [1 0 0], 'LineWidth', 2);        % 红
    h3 = plot(iter, issa, 'Color', [0 1 1], 'LineWidth', 2);       % 青
    h4 = plot(iter, ssa, 'Color', [0 0 1], 'LineWidth', 2);        % 蓝
    h5 = plot(iter, cso, 'Color', [0 0 0], 'LineWidth', 2);        % 黑

    % 设置坐标轴范围和刻度
    xlim([0, 500]);
    ylim([0, y0*1.05]);
    xticks(0:50:500); % 修复横坐标原点重复0
    
    % 移除网格（纯白背景）
    grid off;
    
    % 设置图表整体样式（开放坐标系）- 先设置基本样式
    set(gca, 'FontName', 'TimesSimSun', 'FontSize', 12);
    set(gca, 'LineWidth', 1.2);
    set(gca, 'Box', 'off');  % 关闭边框，创建开放坐标系
    set(gca, 'Color', 'white');  % 确保背景为白色
    
    % 只显示左边和下边的坐标轴线
    set(gca, 'XAxisLocation', 'bottom');
    set(gca, 'YAxisLocation', 'left');
    
    % 设置坐标轴线的颜色和粗细
    set(gca, 'XColor', 'k', 'YColor', 'k');
    set(gca, 'TickDir', 'in');         % 刻度线向内（向上和向右）
    set(gca, 'TickLength', [0.01 0.01]); % 设置刻度线长度，X轴和Y轴一致
    set(gca, 'Position', [0.12, 0.12 0.85, 0.75]);
    
    % 设置坐标轴标签
    xlabel('迭代', 'FontSize', 12, 'FontName', 'TimesSimSun');
    ylabel('适应度', 'FontSize', 12, 'FontName', 'TimesSimSun');
    
    % 设置图例
    legend([h1, h2, h3, h4, h5], {'PSO', 'WOA', 'NEA-SSA', 'SSA', 'AO'}, ...
        'Location', 'northeast', 'FontSize', 12, 'FontName', 'TimesSimSun', 'Box', 'off');

    hold off;

    % 保存高质量图片（论文标准）- 添加纸张设置
    savePath = 'c:\Users\Administrator\Desktop\123\results';
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    
    % 设置纸张属性以避免剪切警告
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [8, 6]);  % 设置合适的纸张尺寸
    
    fileName = 'convergence_comparison';
    
    % PNG格式（高分辨率，白色背景）
    print(gcf, fullfile(savePath, [fileName, '.png']), '-dpng', '-r600');
    
    % EMF格式（Windows矢量图）
    print(gcf, fullfile(savePath, [fileName, '.emf']), '-dmeta', '-r600');
    
    fprintf('收敛对比图已保存至：%s\n', savePath);
    fprintf('文件格式：PNG, EMF\n');
end

function pos = tight_subplot_position(ax)
    % 计算自适应紧凑布局的axes位置，减少留白
    outerpos = get(ax, 'OuterPosition');
    ti = get(ax, 'TightInset');
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    width = outerpos(3) - ti(1) - ti(3);
    height = outerpos(4) - ti(2) - ti(4);
    pos = [left, bottom, width, height];
end