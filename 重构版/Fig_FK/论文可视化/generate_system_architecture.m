function generate_system_architecture()
% 多时间尺度双层优化系统架构示意图
% 简洁水平布局，清晰箭头

    % 创建图形窗口
    figure('Position', [100, 100, 1400, 700], 'Color', 'white');
    hold on;
    axis([0 14 0 10]);
    axis off;
    
    % ========== 上层框架：小时级NEA-SSA多目标优化 ==========
    % 大框（浅灰背景）
    rectangle('Position', [0.3, 5.3, 13.4, 4.0], ...
              'FaceColor', [0.97 0.97 0.97], 'EdgeColor', [0.5 0.5 0.5], ...
              'LineWidth', 2, 'Curvature', 0.08);
    
    % 标题
    text(0.8, 9.0, '小时级NEA-SSA多目标优化', 'FontSize', 14, 'FontWeight', 'bold', ...
         'FontName', 'SimHei', 'Color', [0.2 0.2 0.2]);
    
    % 模块1：光伏/风电预测（橙色）
    rectangle('Position', [0.85, 6.7, 2.2, 1.4], ...
              'FaceColor', [1 0.85 0.6], 'EdgeColor', [0.9 0.6 0.2], ...
              'LineWidth', 2.5, 'Curvature', 0.15);
    text(1.95, 7.5, '光伏/风电', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(1.95, 7.1, '预测', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 箭头→
    quiver(3.1, 7.4, 0.5, 0, 0, 'LineWidth', 3.5, 'Color', 'k', 'MaxHeadSize', 1.2);
    
    % 模块2：负荷预测模块（浅灰）
    rectangle('Position', [3.7, 6.7, 2.2, 1.4], ...
              'FaceColor', [0.88 0.88 0.88], 'EdgeColor', [0.5 0.5 0.5], ...
              'LineWidth', 2.5, 'Curvature', 0.15);
    text(4.8, 7.5, '负荷预测', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(4.8, 7.1, '模块', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 箭头→
    quiver(5.95, 7.4, 0.5, 0, 0, 'LineWidth', 3.5, 'Color', 'k', 'MaxHeadSize', 1.2);
    
    % 模块3：多目标化算法（橙色）
    rectangle('Position', [6.55, 6.7, 2.2, 1.4], ...
              'FaceColor', [1 0.85 0.6], 'EdgeColor', [0.9 0.6 0.2], ...
              'LineWidth', 2.5, 'Curvature', 0.15);
    text(7.65, 7.5, '多目标化', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(7.65, 7.1, '算法', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 箭头→
    quiver(8.8, 7.4, 0.5, 0, 0, 'LineWidth', 3.5, 'Color', 'k', 'MaxHeadSize', 1.2);
    
    % 模块4：多目标优化算法（浅灰）
    rectangle('Position', [9.4, 6.7, 2.2, 1.4], ...
              'FaceColor', [0.88 0.88 0.88], 'EdgeColor', [0.5 0.5 0.5], ...
              'LineWidth', 2.5, 'Curvature', 0.15);
    text(10.5, 7.5, '多目标', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(10.5, 7.1, '优化算法', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % NEA-SSA标签（黑底白字）
    rectangle('Position', [10.0, 8.3, 1.0, 0.35], ...
              'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 1.5, 'Curvature', 0.2);
    text(10.5, 8.475, 'NEA-SSA', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'Times New Roman', 'Color', 'w');
    
    % 箭头→输出
    quiver(11.65, 7.4, 0.5, 0, 0, 'LineWidth', 3.5, 'Color', 'k', 'MaxHeadSize', 1.2);
    
    % 输出框
    text(12.5, 7.75, '优化后：', 'FontSize', 10, 'FontWeight', 'bold', ...
         'FontName', 'SimHei', 'Color', [0.2 0.2 0.2]);
    text(12.5, 7.4, '储能充放电计划', 'FontSize', 9, ...
         'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    text(12.5, 7.1, '电加热器', 'FontSize', 9, ...
         'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    text(12.5, 6.8, '功率指令', 'FontSize', 9, ...
         'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    
    % 上层反馈虚线（简化，只画一条总反馈）
    plot([1.95, 1.95, 10.5, 10.5], [6.65, 6.0, 6.0, 6.65], '--', ...
         'LineWidth', 2, 'Color', [0.6 0.6 0.6]);
    quiver(10.5, 6.0, 0, 0.6, 0, 'LineWidth', 2, 'Color', [0.6 0.6 0.6], 'MaxHeadSize', 2);
    
    % ========== 层间连接 ==========
    % 大箭头（从上层到下层）
    quiver(7, 5.2, 0, -0.6, 0, 'LineWidth', 5, 'Color', [0.5 0.5 0.5], 'MaxHeadSize', 0.9);
    
    % ========== 下层框架：15分钟级NEA-SMC实时控制层 ==========
    % 大框（浅灰背景）
    rectangle('Position', [0.3, 0.5, 13.4, 3.8], ...
              'FaceColor', [0.97 0.97 0.97], 'EdgeColor', [0.5 0.5 0.5], ...
              'LineWidth', 2, 'Curvature', 0.08);
    
    % 标题
    text(0.8, 4.0, '15分钟级NEA-SMC实时控制层', 'FontSize', 14, 'FontWeight', 'bold', ...
         'FontName', 'SimHei', 'Color', [0.2 0.2 0.2]);
    
    % 模块1：状态观测器（蓝色）
    rectangle('Position', [1.2, 1.7, 2.2, 1.4], ...
              'FaceColor', [0.7 0.85 1], 'EdgeColor', [0.2 0.5 0.9], ...
              'LineWidth', 2.5, 'Curvature', 0.15);
    text(2.3, 2.5, '状态观', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(2.3, 2.1, '测器', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 箭头→
    quiver(3.45, 2.4, 0.5, 0, 0, 'LineWidth', 3.5, 'Color', 'k', 'MaxHeadSize', 1.2);
    
    % 模块2：扰动补偿模块（蓝色）
    rectangle('Position', [4.05, 1.7, 2.2, 1.4], ...
              'FaceColor', [0.7 0.85 1], 'EdgeColor', [0.2 0.5 0.9], ...
              'LineWidth', 2.5, 'Curvature', 0.15);
    text(5.15, 2.5, '扰动补偿', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(5.15, 2.1, '模块', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 箭头→
    quiver(6.3, 2.4, 0.5, 0, 0, 'LineWidth', 3.5, 'Color', 'k', 'MaxHeadSize', 1.2);
    
    % 模块3：滑模控制器（浅灰）
    rectangle('Position', [6.9, 1.7, 2.2, 1.4], ...
              'FaceColor', [0.88 0.88 0.88], 'EdgeColor', [0.5 0.5 0.5], ...
              'LineWidth', 2.5, 'Curvature', 0.15);
    text(8.0, 2.5, '滑模控制器', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(8.0, 2.1, '指令', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % NEA-SMC标签（黑底白字）
    rectangle('Position', [7.55, 3.3, 0.9, 0.3], ...
              'FaceColor', 'k', 'EdgeColor', 'k', 'LineWidth', 1.5, 'Curvature', 0.2);
    text(8.0, 3.45, 'NEA-SMC', 'FontSize', 9, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'Times New Roman', 'Color', 'w');
    
    % 箭头→
    quiver(9.15, 2.4, 0.5, 0, 0, 'LineWidth', 3.5, 'Color', 'k', 'MaxHeadSize', 1.2);
    
    % 实时期文字
    text(10.0, 2.4, '实时控制', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    
    % 箭头→设备
    quiver(10.6, 2.4, 0.4, 0, 0, 'LineWidth', 3.5, 'Color', 'k', 'MaxHeadSize', 1.2);
    
    % 设备1：储能系统
    rectangle('Position', [11.1, 2.05, 1.0, 0.75], ...
              'FaceColor', [0.85 0.85 0.85], 'EdgeColor', [0.4 0.4 0.4], ...
              'LineWidth', 2, 'Curvature', 0.1);
    text(11.6, 2.5, '储能', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(11.6, 2.2, '系统', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 设备2：电加热器
    rectangle('Position', [12.2, 2.05, 1.0, 0.75], ...
              'FaceColor', [0.85 0.85 0.85], 'EdgeColor', [0.4 0.4 0.4], ...
              'LineWidth', 2, 'Curvature', 0.1);
    text(12.7, 2.5, '电加', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(12.7, 2.2, '热器', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 下层反馈虚线（简化，一条总反馈）
    plot([2.3, 2.3, 8.0, 8.0], [1.65, 1.1, 1.1, 1.65], '--', ...
         'LineWidth', 2, 'Color', [0.6 0.6 0.6]);
    quiver(8.0, 1.1, 0, 0.5, 0, 'LineWidth', 2, 'Color', [0.6 0.6 0.6], 'MaxHeadSize', 2);
    
    % 实时控制指令说明
    text(11.6, 1.5, '• 实时控制指令：', 'FontSize', 9, ...
         'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    text(11.7, 1.2, '◆ 实时控制', 'FontSize', 9, ...
         'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    
    hold off;
    
    % 保存图片
    savePath = 'results';
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [16, 8]);
    
    fileName = '双层优化系统架构图_彩色';
    
    % PNG格式（高分辨率）
    print(gcf, fullfile(savePath, [fileName, '.png']), '-dpng', '-r600');
    
    % PDF格式（矢量图）
    print(gcf, fullfile(savePath, [fileName, '.pdf']), '-dpdf', '-r600');
    
    % EMF格式（Windows矢量图）
    try
        print(gcf, fullfile(savePath, [fileName, '.emf']), '-dmeta', '-r600');
        fprintf('双层优化系统架构图已保存至：%s\n', savePath);
        fprintf('文件名：%s\n', fileName);
        fprintf('文件格式：PNG, PDF, EMF\n');
    catch
        fprintf('双层优化系统架构图已保存至：%s\n', savePath);
        fprintf('文件名：%s\n', fileName);
        fprintf('文件格式：PNG, PDF\n');
    end
    
    fprintf('图片比例：2:1 (1400×700)\n');
end
