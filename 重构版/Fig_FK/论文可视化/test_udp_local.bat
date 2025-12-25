@echo off
chcp 65001 > nul
echo ====================================
echo UDP本地通信一键测试
echo ====================================
echo.
echo 请选择测试方式：
echo [1] 简化版（JSON文本，推荐）
echo [2] 完整版（二进制数据）
echo [3] 退出
echo.
set /p choice="请输入选项 (1-3): "

if "%choice%"=="1" goto simple
if "%choice%"=="2" goto full
if "%choice%"=="3" goto end

:simple
echo.
echo 正在启动简化版测试...
echo 将打开两个窗口：
echo   - 窗口1: 接收端
echo   - 窗口2: 发送端
echo.
pause
start cmd /k "title UDP接收端 && python udp_simple_receiver.py"
timeout /t 2 > nul
start cmd /k "title UDP发送端 && python udp_simple_sender.py"
echo.
echo 测试已启动！
echo 按任意键关闭本窗口（不影响测试）
pause > nul
goto end

:full
echo.
echo 正在启动完整版测试...
echo 将打开两个窗口：
echo   - 窗口1: 接收端
echo   - 窗口2: 发送端
echo.
pause
start cmd /k "title UDP接收端 && python udp_receiver.py"
timeout /t 2 > nul
start cmd /k "title UDP发送端 && python udp_sender.py"
echo.
echo 测试已启动！
echo 按任意键关闭本窗口（不影响测试）
pause > nul
goto end

:end
exit

