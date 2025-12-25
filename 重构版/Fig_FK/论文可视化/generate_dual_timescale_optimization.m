function generate_dual_timescale_optimization()
% 多时间尺度双层优化流程图
% 彩色配色，科研汇报风格

    % 创建图形窗口（16:9比例）
    figure('Position', [100, 100, 1400, 800], 'Color', 'white');
    hold on;
    axis([0 14 0 10]);
    axis off;
    
    % ========== 标题 ==========
    text(7, 9.7, '多时间尺度双层优化调度框架', 'FontSize', 18, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % ========== 左侧：日前优化层 ==========
    % 大框（蓝色系）
    rectangle('Position', [0.3, 0.3, 6.2, 9], ...
              'FaceColor', [0.90 0.94 0.98], 'EdgeColor', [0.2 0.4 0.7], 'LineWidth', 3);
    
    % 标题
    rectangle('Position', [0.5, 8.5, 5.8, 0.7], ...
              'FaceColor', [0.2 0.4 0.7], 'EdgeColor', [0.1 0.2 0.5], 'LineWidth', 2);
    text(3.4, 8.85, '日前优化层（时间尺度：24h）', 'FontSize', 14, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', 'w');
    
    % 1. 开始
    rectangle('Position', [2.6, 7.8, 1.6, 0.5], ...
              'FaceColor', [0.3 0.7 0.3], 'EdgeColor', [0.2 0.5 0.2], 'LineWidth', 2, 'Curvature', 0.3);
    text(3.4, 8.05, '开始', 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', 'w');
    
    % 箭头：开始→输入数据
    quiver(3.4, 7.75, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 2. 输入数据（菱形）
    patch([3.4-0.9, 3.4, 3.4+0.9, 3.4], [7.2, 7.5, 7.2, 6.9], ...
          [0.95 0.95 0.7], 'EdgeColor', [0.7 0.7 0.3], 'LineWidth', 2);
    text(3.4, 7.2, '输入数据', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 左侧数据框
    rectangle('Position', [0.6, 6.2, 1.8, 1.5], ...
              'FaceColor', [1 0.95 0.8], 'EdgeColor', [0.8 0.6 0.2], 'LineWidth', 1.5);
    text(1.5, 7.1, '系统数据加载:', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(1.5, 6.75, '光电/风电/负荷', 'FontSize', 9, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimSun');
    text(1.5, 6.5, '预测曲线、电价', 'FontSize', 9, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimSun');
    text(1.5, 6.25, '设备参数', 'FontSize', 9, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimSun');
    
    % 右侧参数框
    rectangle('Position', [4.2, 6.2, 1.8, 1.5], ...
              'FaceColor', [0.9 0.95 1], 'EdgeColor', [0.3 0.5 0.8], 'LineWidth', 1.5);
    text(5.1, 7.1, '算法参数设置:', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    text(5.1, 6.75, 'SSA参数、', 'FontSize', 9, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimSun');
    text(5.1, 6.5, 'SMC参数', 'FontSize', 9, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimSun');
    
    % 数据输入箭头
    quiver(2.5, 7.2, 0.4, 0, 0, 'LineWidth', 2, 'Color', [0.7 0.5 0.1], 'MaxHeadSize', 2);
    quiver(4.1, 7.2, -0.3, 0, 0, 'LineWidth', 2, 'Color', [0.3 0.5 0.8], 'MaxHeadSize', 2);
    
    % 箭头：输入数据→神经网络
    quiver(3.4, 6.85, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 3. 神经网络智能初始化种群
    rectangle('Position', [2.0, 6.0, 2.8, 0.5], ...
              'FaceColor', [0.95 0.85 0.95], 'EdgeColor', [0.6 0.3 0.7], 'LineWidth', 2);
    text(3.4, 6.25, '神经网络智能初始化种群', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.5 0.2 0.6]);
    
    % 箭头：神经网络→SSA迭代
    quiver(3.4, 5.95, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 4. SSA迭代优化（菱形，强调）
    patch([3.4-1.0, 3.4, 3.4+1.0, 3.4], [5.4, 5.7, 5.4, 5.1], ...
          [1 0.85 0.7], 'EdgeColor', [0.8 0.4 0.2], 'LineWidth', 2.5);
    text(3.4, 5.4, 'SSA迭代寻优', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.6 0.2 0]);
    
    % 箭头：SSA→参数自适应
    quiver(3.4, 5.05, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 5. 参数自适应
    rectangle('Position', [2.0, 4.3, 2.8, 0.5], ...
              'FaceColor', [0.85 0.95 1], 'EdgeColor', [0.2 0.5 0.8], 'LineWidth', 2);
    text(3.4, 4.55, '参数自适应（神经网络）', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.1 0.3 0.6]);
    
    % 箭头→个体进化
    quiver(3.4, 4.25, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 6. 个体进化
    rectangle('Position', [2.2, 3.6, 2.4, 0.5], ...
              'FaceColor', [0.9 1 0.9], 'EdgeColor', [0.3 0.7 0.3], 'LineWidth', 2);
    text(3.4, 3.85, '个体进化', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.2 0.5 0.2]);
    
    % 箭头→多目标
    quiver(3.4, 3.55, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 7. 多目标适应度评估
    rectangle('Position', [2.0, 2.85, 2.8, 0.5], ...
              'FaceColor', [1 0.9 0.8], 'EdgeColor', [0.8 0.5 0.2], 'LineWidth', 2);
    text(3.4, 3.1, '多目标适应度评估', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.6 0.3 0]);
    
    % 箭头→判断
    quiver(3.4, 2.8, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 8. 判断收敛（菱形）
    patch([3.4-1.0, 3.4, 3.4+1.0, 3.4], [2.2, 2.5, 2.2, 1.9], ...
          [1 1 0.85], 'EdgeColor', [0.8 0.6 0.1], 'LineWidth', 2.5);
    text(3.4, 2.2, '满足迭代停止条件?', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 否→循环回SSA
    plot([2.3, 0.9, 0.9, 2.3], [2.2, 2.2, 5.4, 5.4], '--', 'LineWidth', 2.5, 'Color', [0.8 0.3 0.3]);
    quiver(2.3, 5.4, 0.15, 0, 0, 'LineWidth', 2.5, 'Color', [0.8 0.3 0.3], 'MaxHeadSize', 1.5);
    text(1.1, 3.8, '否', 'FontSize', 11, 'FontWeight', 'bold', ...
         'FontName', 'SimHei', 'Color', [0.8 0.3 0.3]);
    
    % 是→输出
    quiver(3.4, 1.85, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    text(3.7, 1.7, '是', 'FontSize', 11, 'FontWeight', 'bold', ...
         'FontName', 'SimHei', 'Color', [0.2 0.6 0.2]);
    
    % 9. 输出结果
    rectangle('Position', [1.8, 0.85, 3.2, 0.5], ...
              'FaceColor', [0.7 0.9 0.7], 'EdgeColor', [0.2 0.6 0.2], 'LineWidth', 2);
    text(3.4, 1.1, '输出：最优日前调度计划', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.1 0.4 0.1]);
    
    % ========== 右侧：日内调度层 ==========
    % 大框（红色系）
    rectangle('Position', [7.5, 0.3, 6.2, 9], ...
              'FaceColor', [0.98 0.93 0.93], 'EdgeColor', [0.8 0.3 0.3], 'LineWidth', 3);
    
    % 标题
    rectangle('Position', [7.7, 8.5, 5.8, 0.7], ...
              'FaceColor', [0.8 0.3 0.3], 'EdgeColor', [0.6 0.2 0.2], 'LineWidth', 2);
    text(10.6, 8.85, '日内调度层（时间尺度：15min）', 'FontSize', 14, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', 'w');
    
    % 1. 获取计划
    rectangle('Position', [9.0, 7.65, 3.2, 0.5], ...
              'FaceColor', [0.7 0.9 0.7], 'EdgeColor', [0.2 0.6 0.2], 'LineWidth', 2);
    text(10.6, 7.9, '获取上层日前计划', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.1 0.4 0.1]);
    
    % 箭头→控制器
    quiver(10.6, 7.6, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 2. NEA-SMC控制器（菱形，强调）
    patch([10.6-1.2, 10.6, 10.6+1.2, 10.6], [7.0, 7.35, 7.0, 6.65], ...
          [1 0.8 0.7], 'EdgeColor', [0.8 0.3 0.2], 'LineWidth', 2.5);
    text(10.6, 7.1, '下层NEA-SMC控制器', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.6 0.2 0]);
    text(10.6, 6.85, '实时调整', 'FontSize', 10, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.6 0.2 0]);
    
    % 箭头→获取状态
    quiver(10.6, 6.6, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 3. 获取状态
    rectangle('Position', [8.9, 5.85, 3.4, 0.5], ...
              'FaceColor', [0.85 0.9 1], 'EdgeColor', [0.2 0.4 0.8], 'LineWidth', 2);
    text(10.6, 6.1, '获取当前状态与上层计划', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.1 0.2 0.6]);
    
    % 箭头→滑模面
    quiver(10.6, 5.8, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 4. 滑模面参数
    rectangle('Position', [9.1, 5.15, 3.0, 0.5], ...
              'FaceColor', [0.95 0.85 1], 'EdgeColor', [0.5 0.2 0.8], 'LineWidth', 2);
    text(10.6, 5.4, '滑模面参数自适应', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.4 0.1 0.6]);
    
    % 箭头→边界层
    quiver(10.6, 5.1, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 5. 边界层
    rectangle('Position', [9.1, 4.45, 3.0, 0.5], ...
              'FaceColor', [1 0.95 0.85], 'EdgeColor', [0.8 0.6 0.2], 'LineWidth', 2);
    text(10.6, 4.7, '边界层厚度自适应', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.6 0.4 0]);
    
    % 箭头→扰动观测
    quiver(10.6, 4.4, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 6. 扰动观测
    rectangle('Position', [8.9, 3.75, 3.4, 0.5], ...
              'FaceColor', [0.9 1 0.95], 'EdgeColor', [0.2 0.7 0.4], 'LineWidth', 2);
    text(10.6, 4.0, '扰动观测与补偿自适应', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.1 0.5 0.2]);
    
    % 箭头→控制输入
    quiver(10.6, 3.7, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 7. 控制输入
    rectangle('Position', [9.0, 3.05, 3.2, 0.5], ...
              'FaceColor', [1 0.85 0.85], 'EdgeColor', [0.8 0.3 0.3], 'LineWidth', 2);
    text(10.6, 3.3, '计算控制输入并执行', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.6 0.2 0.2]);
    
    % 箭头→时间判断
    quiver(10.6, 3.0, 0, -0.3, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    
    % 8. 时间结束判断（菱形）
    patch([10.6-0.9, 10.6, 10.6+0.9, 10.6], [2.4, 2.7, 2.4, 2.1], ...
          [1 1 0.9], 'EdgeColor', [0.7 0.5 0.2], 'LineWidth', 2.5);
    text(10.6, 2.4, '时间结束', 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei');
    
    % 否→循环回控制器
    plot([11.6, 13.0, 13.0, 11.6], [2.4, 2.4, 7.0, 7.0], '--', 'LineWidth', 2.5, 'Color', [0.8 0.3 0.3]);
    quiver(11.6, 7.0, -0.15, 0, 0, 'LineWidth', 2.5, 'Color', [0.8 0.3 0.3], 'MaxHeadSize', 1.5);
    text(12.8, 4.7, '否', 'FontSize', 11, 'FontWeight', 'bold', ...
         'FontName', 'SimHei', 'Color', [0.8 0.3 0.3]);
    
    % 是→结束
    quiver(10.6, 2.05, 0, -0.5, 0, 'LineWidth', 2.5, 'Color', 'k', 'MaxHeadSize', 1.5);
    text(10.9, 1.8, '是', 'FontSize', 11, 'FontWeight', 'bold', ...
         'FontName', 'SimHei', 'Color', [0.2 0.6 0.2]);
    
    % 9. 结束
    rectangle('Position', [9.8, 0.85, 1.6, 0.5], ...
              'FaceColor', [0.3 0.7 0.3], 'EdgeColor', [0.2 0.5 0.2], 'LineWidth', 2, 'Curvature', 0.3);
    text(10.6, 1.1, '结束', 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', 'w');
    
    % ========== 层间连接 ==========
    % 日前→日内的大箭头
    quiver(5.1, 1.1, 2.2, 6.7, 0, 'LineWidth', 4, 'Color', [0.3 0.6 0.3], 'MaxHeadSize', 0.6);
    
    % 连接标签
    rectangle('Position', [6.0, 4.5, 2.5, 0.6], ...
              'FaceColor', [0.95 1 0.95], 'EdgeColor', [0.3 0.6 0.3], 'LineWidth', 2.5);
    text(7.25, 4.8, '日前计划传递', 'FontSize', 11, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'SimHei', 'Color', [0.2 0.5 0.2]);
    
    % 添加时间尺度说明（左下角）
    rectangle('Position', [0.5, 0.5, 2.8, 0.25], ...
              'FaceColor', [0.9 0.9 0.9], 'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1);
    text(1.9, 0.625, '时间分辨率: 1小时', 'FontSize', 9, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimSun', 'Color', [0.3 0.3 0.3]);
    
    % 添加时间尺度说明（右下角）
    rectangle('Position', [11.0, 0.5, 2.8, 0.25], ...
              'FaceColor', [0.9 0.9 0.9], 'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1);
    text(12.4, 0.625, '时间分辨率: 15分钟', 'FontSize', 9, ...
         'HorizontalAlignment', 'center', 'FontName', 'SimSun', 'Color', [0.3 0.3 0.3]);
    
    hold off;
    
    % 保存图片
    savePath = 'results';
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [16, 9]);
    
    fileName = '多时间尺度双层优化流程图_彩色';
    
    % PNG格式（高分辨率）
    print(gcf, fullfile(savePath, [fileName, '.png']), '-dpng', '-r600');
    
    % PDF格式（矢量图）
    print(gcf, fullfile(savePath, [fileName, '.pdf']), '-dpdf', '-r600');
    
    % EMF格式（Windows矢量图）
    try
        print(gcf, fullfile(savePath, [fileName, '.emf']), '-dmeta', '-r600');
        fprintf('多时间尺度双层优化流程图已保存至：%s\n', savePath);
        fprintf('文件名：%s\n', fileName);
        fprintf('文件格式：PNG, PDF, EMF\n');
    catch
        fprintf('多时间尺度双层优化流程图已保存至：%s\n', savePath);
        fprintf('文件名：%s\n', fileName);
        fprintf('文件格式：PNG, PDF\n');
    end
    
    fprintf('图片比例：16:9 (1400×800)\n');
end

