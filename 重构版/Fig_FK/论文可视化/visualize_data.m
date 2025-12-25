function paper_visualization_fixed()
% 修正版论文级数据可视化脚本
% 功能：生成符合论文标准的清洁风格图表
% 修正：中文显示、背景样式、风电标签

    % 清空工作空间
    clear; clc; close all;
    
    % 数据文件路径
    dataPath = 'C:\Users\Administrator\Desktop\备份\论文版\论文可视化';
    hourlyFile = fullfile(dataPath, 'data_2021_hourly.mat');
    
    % 加载数据
    fprintf('正在加载数据...\n');
    load(hourlyFile);   % 加载 data_hr
    
    % 用户输入起始日期（默认使用典型日期）
    fprintf('请输入起始日期（格式：yyyy-mm-dd，直接回车使用默认日期2021-05-05）：\n');
    dateInput = input('', 's');
    
    if isempty(dateInput)
        startDate = datetime('2021-05-05', 'InputFormat', 'yyyy-MM-dd');
        fprintf('使用默认日期：2021-05-05\n');
    else
        try
            startDate = datetime(dateInput, 'InputFormat', 'yyyy-MM-dd');
        catch
            warning('日期格式错误，使用默认日期：2021-04-05');
            startDate = datetime('2021-04-05', 'InputFormat', 'yyyy-MM-dd');
        end
    end
    
    % 计算结束日期（25小时后，包含第二天0点）
    endDate = startDate + hours(25);
    
    % 筛选25小时数据（包含第二天0点）
    hourlyMask = (data_hr.Timestamp >= startDate) & (data_hr.Timestamp < endDate);
    hourlyData = data_hr(hourlyMask, :);
    
    % 创建时间轴（0-24小时，包含第24小时）
    timeHours = 0:24;
    
    % 提取数据（确保有25个数据点：0-24小时）
    windPower = hourlyData.Wind_MW(1:25);  % 取前25个数据点
    solarPower = hourlyData.Solar_MW(1:25);
    loadDemand = hourlyData.Load_MW(1:25);
    
    % 创建论文级图表（修正版）- 调整图表尺寸
    figure('Position', [220, 220, 600, 400], 'Color', 'white');
    
    % 设置图表属性
    hold on;
    
    % 绘制曲线（使用论文标准样式）
    h1 = plot(timeHours, loadDemand, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, ...
              'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');
    h2 = plot(timeHours, solarPower, 'g-s', 'LineWidth', 1.5, 'MarkerSize', 4, ...
              'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g');
    h3 = plot(timeHours, windPower, 'b-^', 'LineWidth', 1.5, 'MarkerSize', 4, ...
              'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
    
    % 设置坐标轴范围和刻度
    xlim([0, 24]);           % 确保X轴范围是0-24
    xticks(0:2:24);          % X轴刻度：0, 2, 4, ..., 24
    ylim([0, 40]);           % 固定Y轴范围是0-40
    yticks(0:5:40);          % Y轴刻度：0, 5, 10, 15, 20, 25, 30, 35, 40
   
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
    set(gca, 'TickDir', 'in');         % 刻度线向内（向上和向右）
    set(gca, 'TickLength', [0.01 0.01]); % 设置刻度线长度，X轴和Y轴一致
    set(gca, 'Position', [0.1, 0.15, 0.85, 0.75]);
    
    % 设置坐标轴标签
    xlabel('时间 (h)', 'FontSize', 12, 'FontName', 'TimesSimSun');
    ylabel('功率 (MW)', 'FontSize', 12, 'FontName', 'TimesSimSun');
    
    % 设置图例
    legend([h1, h2, h3], {'负荷', '光伏', '风电'}, ...
           'Location', 'northwest', 'FontSize', 12, 'FontName', 'TimesSimSun', ...
           'Box', 'off');
    
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
    
    % 保存多种格式
    fileName = sprintf('paper_figure_clean_%s', datestr(startDate, 'yyyymmdd'));
    
    % PNG格式（高分辨率，白色背景）
    print(gcf, fullfile(savePath, [fileName, '.png']), '-dpng', '-r600');

    
    % EMF格式（Windows矢量图）
    print(gcf, fullfile(savePath, [fileName, '.emf']), '-dmeta', '-r600');
    
    fprintf('修正版图表已保存至：%s\n', savePath);
    fprintf('文件格式：PNG, EPS, PDF, EMF\n');
    
    
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