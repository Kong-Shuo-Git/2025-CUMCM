#!/bin/bash

# 碳化硅外延层红外多光束干涉测量系统 - Python可视化启动脚本

echo "🔬 碳化硅外延层红外多光束干涉测量系统 - Python可视化"
echo "=================================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ 错误: 未找到Python环境，请安装Python 3.8或更高版本"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python环境检查通过"

# 检查并安装依赖
echo "📦 正在检查Python依赖..."
if [ -f "requirements.txt" ]; then
    $PYTHON_CMD -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "⚠️  警告: 部分依赖安装失败，程序可能无法正常运行"
    fi
else
    echo "⚠️  警告: 未找到requirements.txt文件"
fi

echo "✅ 依赖检查完成"

# 启动Python可视化程序
echo "🚀 正在启动Python可视化程序..."
echo "📊 图形界面将在新窗口中打开"
echo "⏹️  关闭窗口退出程序"
echo ""

$PYTHON_CMD Simulation.py