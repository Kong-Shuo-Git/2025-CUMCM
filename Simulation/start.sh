#!/bin/bash

# 碳化硅外延层红外多光束干涉测量系统启动脚本

echo "🔬 碳化硅外延层红外多光束干涉测量系统"
echo "=================================="

# 检查Java环境
if ! command -v java &> /dev/null; then
    echo "❌ 错误: 未找到Java环境，请安装Java 17或更高版本"
    exit 1
fi

# 检查Maven环境
if ! command -v mvn &> /dev/null; then
    echo "❌ 错误: 未找到Maven环境，请安装Maven 3.6或更高版本"
    exit 1
fi

echo "✅ 环境检查通过"

# 编译项目
echo "📦 正在编译项目..."
mvn clean compile

if [ $? -ne 0 ]; then
    echo "❌ 项目编译失败"
    exit 1
fi

echo "✅ 项目编译成功"

# 启动Spring Boot应用
echo "🚀 正在启动Web应用..."
echo "📱 访问地址: http://localhost:8080"
echo "⏹️  按 Ctrl+C 停止应用"
echo ""

mvn spring-boot:run