function cost_analysis()
    %% 直接填写各项成本数据（单位：元）
    % 传统策略
    ch_cost_before = 12667;   % 储能成本
    grid_cost_before = 15466;  % 电网成本
    total_cost_before = 28133; % 总成本
    
    % 优化策略
    ch_cost_after = 8498;    % 储能成本
    grid_cost_after = 4089;   % 电网成本
    total_cost_after = 12587; % 总成本
    
    % 成本分类（中文）
    categories = {'储能成本', '电网成本', '总成本'};
    before_costs = [ch_cost_before, grid_cost_before, total_cost_before];
    after_costs = [ch_cost_after, grid_cost_after, total_cost_after];
    
    %% 绘制成本对比图
    figure('Position', [180, 180, 500, 350], 'Color', 'white');
    x = 1:length(categories);
    width = 0.35;
    hold on;
    b1 = bar(x - width/2, before_costs, width, 'FaceColor', [0.85, 0.33, 0.33], ...
             'EdgeColor', [0.2, 0.2, 0.2], 'LineWidth', 1.2);
    b2 = bar(x + width/2, after_costs, width, 'FaceColor', [0.33, 0.66, 0.33], ...
             'EdgeColor', [0.2, 0.2, 0.2], 'LineWidth', 1.2);
    
    % 设置坐标轴范围和刻度
    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', categories);
    ylim([0, max([before_costs, after_costs]) * 1.15]);
    
    % 移除网格（纯白背景）
    grid off;
    
    % 设置图表整体样式（开放坐标系）- 先设置基本样式
    set(gca, 'FontSize', 12, 'FontName', 'TimesSimSun');
    set(gca, 'LineWidth', 1.2);
    set(gca, 'Box', 'off');  % 关闭边框，创建开放坐标系
    set(gca, 'Color', 'white');  % 确保背景为白色
    
    % 只显示左边和下边的坐标轴线
    set(gca, 'XAxisLocation', 'bottom');
    set(gca, 'YAxisLocation', 'left');
    
    % 设置坐标轴线的颜色和粗细
    set(gca, 'XColor', 'k', 'YColor', 'k');
    set(gca, 'LineWidth', 1.2);
    
    % 统一设置刻度线样式（向内的小凸起）
    set(gca, 'TickDir', 'in');         % 刻度线向内（向上和向右）
    set(gca, 'TickLength', [0.01 0.01]); % 设置刻度线长度，X轴和Y轴一致
    set(gca, 'Position', [0.1, 0.15, 0.85, 0.75]);
    
    % 设置坐标轴标签
    ylabel('成本（元）', 'FontSize', 12, 'FontName', 'TimesSimSun');
    
    % 设置图例
    legend([b1, b2], {'传统策略', '优化策略'}, ...
           'Location', 'northwest', 'FontSize', 12, 'FontName', 'TimesSimSun', ...
           'Box', 'off');
    
    % 数值标签
    for i = 1:length(categories)
        text(i - width/2, before_costs(i) + max([before_costs, after_costs])*0.03, ...
             sprintf('%d', round(before_costs(i))), 'HorizontalAlignment', 'center', ...
             'FontSize', 12, 'FontName', 'TimesSimSun');
        text(i + width/2, after_costs(i) + max([before_costs, after_costs])*0.03, ...
             sprintf('%d', round(after_costs(i))), 'HorizontalAlignment', 'center', ...
             'FontSize', 12, 'FontName', 'TimesSimSun');
    end
    hold off;
    
    % 保存高质量图表（论文标准）- 添加纸张设置
    savePath = 'c:\Users\Administrator\Desktop\123\results';
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    
    % 设置纸张属性以避免剪切警告
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [8, 6]);  % 设置合适的纸张尺寸
    
    fileName = 'cost_optimization_comparison_20210505';
    
    % PNG格式（高分辨率，白色背景）
    print(gcf, fullfile(savePath, [fileName, '.png']), '-dpng', '-r600');
    
    % EMF格式（Windows矢量图）
    print(gcf, fullfile(savePath, [fileName, '.emf']), '-dmeta', '-r600');
    
    fprintf('成本对比图表已保存至：%s\n', savePath);
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