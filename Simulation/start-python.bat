@echo off
chcp 65001 >nul

REM 碳化硅外延层红外多光束干涉测量系统 - Python可视化启动脚本 (Windows)

echo 🔬 碳化硅外延层红外多光束干涉测量系统 - Python可视化
echo ==================================================

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ 错误: 未找到Python环境，请安装Python 3.8或更高版本
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

echo ✅ Python环境检查通过

REM 检查并安装依赖
echo 📦 正在检查Python依赖...
if exist requirements.txt (
    %PYTHON_CMD% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ⚠️  警告: 部分依赖安装失败，程序可能无法正常运行
    )
) else (
    echo ⚠️  警告: 未找到requirements.txt文件
)

echo ✅ 依赖检查完成

REM 启动Python可视化程序
echo 🚀 正在启动Python可视化程序...
echo 📊 图形界面将在新窗口中打开
echo ⏹️  关闭窗口退出程序
echo.

%PYTHON_CMD% Simulation.py

pause