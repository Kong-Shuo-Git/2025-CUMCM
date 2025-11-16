@echo off
chcp 65001 >nul

REM 碳化硅外延层红外多光束干涉测量系统启动脚本 (Windows)

echo 🔬 碳化硅外延层红外多光束干涉测量系统
echo ==================================

REM 检查Java环境
java -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Java环境，请安装Java 17或更高版本
    pause
    exit /b 1
)

REM 检查Maven环境
mvn -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Maven环境，请安装Maven 3.6或更高版本
    pause
    exit /b 1
)

echo ✅ 环境检查通过

REM 编译项目
echo 📦 正在编译项目...
call mvn clean compile

if %errorlevel% neq 0 (
    echo ❌ 项目编译失败
    pause
    exit /b 1
)

echo ✅ 项目编译成功

REM 启动Spring Boot应用
echo 🚀 正在启动Web应用...
echo 📱 访问地址: http://localhost:8080
echo ⏹️  按 Ctrl+C 停止应用
echo.

call mvn spring-boot:run