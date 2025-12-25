function generate_hardware_deployment()
% Learning MPC系统硬件部署架构示意图
% 四角星形布局，交换机居中

    % 创建图形窗口
    figure('Position', [100, 100, 1200, 900], 'Color', 'white');
    hold on;
    axis([0 12 0 10]);
    axis off;
    
    % ========== 标题 ==========
    text(6, 9.5, 'Learning-MPC风光熔盐储能系统半实物硬件部署示意图', ...
         'FontSize', 18, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % ========== 左上角：PC-1 上位机（蓝色）==========
    % 显示器外框阴影
    rectangle('Position', [0.55, 7.05, 2.6, 1.8], ...
              'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none', 'Curvature', 0.05);
    % 显示器外框
    rectangle('Position', [0.5, 7.1, 2.6, 1.8], ...
              'FaceColor', [0.2 0.4 0.8], 'EdgeColor', [0.1 0.3 0.6], ...
              'LineWidth', 2.5, 'Curvature', 0.05);
    % 屏幕
    rectangle('Position', [0.65, 7.3, 2.3, 1.4], ...
              'FaceColor', [0.1 0.2 0.3], 'EdgeColor', [0.3 0.5 0.7], 'LineWidth', 1.5);
    % 屏幕内容（曲线）
    t_pc1 = linspace(0.8, 2.8, 40);
    y_pc1 = 8.1 + 0.3*sin(2*pi*(t_pc1-0.8)/2);
    plot(t_pc1, y_pc1, '-', 'LineWidth', 2, 'Color', [0.2 1 0.2]);
    
    % 显示器底座
    rectangle('Position', [1.2, 6.9, 1.2, 0.15], ...
              'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.5);
    
    % 文字标签（正下方）
    text(1.8, 6.6, 'PC-1 上位机', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.2 0.4 0.8]);
    text(1.8, 6.3, 'IP: 192.168.1.10', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.5 0.5 0.5]);
    
    % ========== 右上角：工控机（红色，核心）==========
    % 机箱阴影
    rectangle('Position', [8.75, 7.05, 2.8, 1.8], ...
              'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none', 'Curvature', 0.08);
    % 机箱主体
    rectangle('Position', [8.7, 7.1, 2.8, 1.8], ...
              'FaceColor', [0.95 0.7 0.7], 'EdgeColor', [0.8 0.2 0.2], ...
              'LineWidth', 3, 'Curvature', 0.08);
    
    % 指示灯
    rectangle('Position', [8.9, 8.65, 0.15, 0.15], ...
              'FaceColor', [0 1 0], 'EdgeColor', 'k', 'LineWidth', 0.5, 'Curvature', 1);
    rectangle('Position', [9.15, 8.65, 0.15, 0.15], ...
              'FaceColor', [1 0.8 0], 'EdgeColor', 'k', 'LineWidth', 0.5, 'Curvature', 1);
    
    % 机箱面板显示
    text(10.1, 8.3, '工控机', 'FontSize', 13, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.7 0.1 0.1]);
    text(10.1, 7.9, 'Learning MPC', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'Times New Roman', 'Color', [0.5 0.1 0.1]);
    text(10.1, 7.5, '实时优化', 'FontSize', 10, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimSun', 'Color', [0.3 0.3 0.3]);
    
    % 文字标签（正下方）
    text(10.1, 6.8, '工控机', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.8 0.2 0.2]);
    text(10.1, 6.5, 'Learning MPC算法核心', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    text(10.1, 6.25, 'IP: 192.168.1.100', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.5 0.5 0.5]);
    
    % ========== 中心：以太网交换机（灰色）==========
    % 阴影
    rectangle('Position', [4.85, 4.45, 2.3, 1.5], ...
              'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none', 'Curvature', 0.1);
    % 主框
    rectangle('Position', [4.8, 4.5, 2.3, 1.5], ...
              'FaceColor', [0.4 0.4 0.4], 'EdgeColor', [0.2 0.2 0.2], ...
              'LineWidth', 3, 'Curvature', 0.1);
    % 端口指示灯（上下两排）
    for i = 0:4
        rectangle('Position', [5.0 + i*0.35, 4.7, 0.25, 0.15], ...
                  'FaceColor', [0.2 0.8 0.2], 'EdgeColor', 'k', 'LineWidth', 1);
        rectangle('Position', [5.0 + i*0.35, 5.6, 0.25, 0.15], ...
                  'FaceColor', [0.2 0.8 0.2], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    
   
    
    % 文字标签（正下方）
    text(5.95, 4.2, '交换机', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    text(5.95, 3.95, ' udp通信 ', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.5 0.5 0.5]);
    
    % ========== 左下角：PC-2 数据采集（绿色）==========
    % 显示器外框阴影
    rectangle('Position', [0.55, 1.55, 2.6, 1.8], ...
              'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none', 'Curvature', 0.05);
    % 显示器外框
    rectangle('Position', [0.5, 1.6, 2.6, 1.8], ...
              'FaceColor', [0.2 0.7 0.2], 'EdgeColor', [0.1 0.5 0.1], ...
              'LineWidth', 2.5, 'Curvature', 0.05);
    % 屏幕（黑色）
    rectangle('Position', [0.65, 1.8, 2.3, 1.4], ...
              'FaceColor', [0.1 0.2 0.1], 'EdgeColor', [0.3 0.6 0.3], 'LineWidth', 1.5);
    % 屏幕内容（绿色终端文字）
    text(1.8, 3.0, '风电: 22.3 MW', 'FontSize', 9, 'FontName', 'SimHei', ...
         'Color', [0.3 1 0.3], 'HorizontalAlignment', 'center');
    text(1.8, 2.7, '光伏: 15.2 MW', 'FontSize', 9, 'FontName', 'SimHei', ...
         'Color', [0.3 1 0.3], 'HorizontalAlignment', 'center');
    text(1.8, 2.4, '负荷: 28.5 MW', 'FontSize', 9, 'FontName', 'SimHei', ...
         'Color', [0.3 1 0.3], 'HorizontalAlignment', 'center');
    text(1.8, 2.1, 'SOC:  55.2 %', 'FontSize', 9, 'FontName', 'SimHei', ...
         'Color', [0.3 1 0.3], 'HorizontalAlignment', 'center');
    
    % 显示器底座
    rectangle('Position', [1.2, 1.4, 1.2, 0.15], ...
              'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.5);
    
    % 文字标签（正下方）
    text(1.8, 1.1, 'PC-2 仿真模型', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.2 0.6 0.2]);
    text(1.8, 0.8, '数据模拟', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    text(1.8, 0.55, 'IP: 192.168.1.20', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.5 0.5 0.5]);
    
    % ========== 右下角：PC-3 执行器（橙色）==========
    % 显示器外框阴影
    rectangle('Position', [8.95, 1.55, 2.6, 1.8], ...
              'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none', 'Curvature', 0.05);
    % 显示器外框
    rectangle('Position', [8.9, 1.6, 2.6, 1.8], ...
              'FaceColor', [0.9 0.5 0.2], 'EdgeColor', [0.7 0.3 0.1], ...
              'LineWidth', 2.5, 'Curvature', 0.05);
    % 屏幕（黑色）
    rectangle('Position', [9.05, 1.8, 2.3, 1.4], ...
              'FaceColor', [0.2 0.15 0.1], 'EdgeColor', [0.7 0.5 0.2], 'LineWidth', 1.5);
    % 屏幕内容（橙色终端文字）
    text(10.2, 3.0, '储能: -7.5 MW', 'FontSize', 9, 'FontName', 'SimHei', ...
         'Color', [1 0.8 0.3], 'HorizontalAlignment', 'center');
    text(10.2, 2.7, '加热: +8.2 MW', 'FontSize', 9, 'FontName', 'SimHei', ...
         'Color', [1 0.8 0.3], 'HorizontalAlignment', 'center');
    text(10.2, 2.4, '电网: +0.5 MW', 'FontSize', 9, 'FontName', 'SimHei', ...
         'Color', [1 0.8 0.3], 'HorizontalAlignment', 'center');
    text(10.2, 2.1, '状态: 执行中', 'FontSize', 9, 'FontName', 'SimHei', ...
         'Color', [0.3 1 0.3], 'HorizontalAlignment', 'center');
    
    % 显示器底座
    rectangle('Position', [9.6, 1.4, 1.2, 0.15], ...
              'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.5);
    
    % 文字标签（正下方）
    text(10.2, 1.1, 'PC-3 执行器', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.8 0.4 0.1]);
    text(10.2, 0.8, '设备控制', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);
    text(10.2, 0.55, 'IP: 192.168.1.30', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.5 0.5 0.5]);
    

    
    hold off;
    
    % 保存图片
    savePath = 'results';
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [14, 10]);
    
    fileName = '硬件部署架构图_彩色';
    
    % PNG格式（高分辨率）
    print(gcf, fullfile(savePath, [fileName, '.png']), '-dpng', '-r600');
    
    % PDF格式（矢量图）
    print(gcf, fullfile(savePath, [fileName, '.pdf']), '-dpdf', '-r600');
    
    % EMF格式（Windows矢量图）
    try
        print(gcf, fullfile(savePath, [fileName, '.emf']), '-dmeta', '-r600');
        fprintf('硬件部署架构图已保存至：%s\n', savePath);
        fprintf('文件名：%s\n', fileName);
        fprintf('文件格式：PNG, PDF, EMF\n');
    catch
        fprintf('硬件部署架构图已保存至：%s\n', savePath);
        fprintf('文件名：%s\n', fileName);
        fprintf('文件格式：PNG, PDF\n');
    end
    
    fprintf('图片比例：4:3 (1200×900)\n');
end

