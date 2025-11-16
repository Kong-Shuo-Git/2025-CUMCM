#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025数学建模竞赛项目快速运行脚本
一键运行所有解决方案和灵敏度分析
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    """打印项目横幅"""
    print("=" * 80)
    print("    2025年全国大学生数学建模竞赛")
    print("    碳化硅外延层厚度红外多光束干涉分析")
    print("=" * 80)
    print()

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ 错误: 需要Python 3.7或更高版本")
        return False
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = ['numpy', 'pandas', 'scipy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 检查数据文件
    data_files = [
        "附件1_processed.xlsx",
        "附件2_processed.xlsx", 
        "附件3_processed.xlsx",
        "附件4_processed.xlsx"
    ]
    
    missing_files = []
    for file in data_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"❌ 数据文件缺失: {file}")
        else:
            print(f"✅ 数据文件存在: {file}")
    
    if missing_files:
        print(f"\n⚠️  缺少数据文件: {', '.join(missing_files)}")
        print("请确保所有数据文件都在当前目录下")
        return False
    
    print("✅ 环境检查完成\n")
    return True

def run_script(script_name, description):
    """运行单个Python脚本"""
    print(f"🚀 运行 {description}...")
    print(f"   脚本: {script_name}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5分钟超时
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} 运行成功 (耗时: {elapsed_time:.1f}秒)")
            if result.stdout:
                print("   输出:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print(f"❌ {description} 运行失败")
            print("   错误:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 运行超时 (5分钟)")
        return False
    except Exception as e:
        print(f"❌ {description} 运行异常: {str(e)}")
        return False
    
    print()
    return True

def run_all_solutions():
    """运行所有主解决方案"""
    print("📊 开始运行主解决方案...")
    print()
    
    solutions = [
        ("Problem2_solution.py", "问题2 - 碳化硅外延层厚度分析"),
        ("Problem3_solution.py", "问题3 - 硅外延层厚度分析"),
        ("Problem4_solution.py", "问题4 - SiC多波束干涉综合分析")
    ]
    
    success_count = 0
    for script, desc in solutions:
        if os.path.exists(script):
            if run_script(script, desc):
                success_count += 1
        else:
            print(f"⚠️  脚本不存在: {script}")
    
    print(f"📈 主解决方案完成: {success_count}/{len(solutions)} 个成功\n")
    return success_count == len(solutions)

def run_sensitivity_analysis():
    """运行所有灵敏度分析"""
    print("🔬 开始运行灵敏度分析...")
    print()
    
    sensitivity_scripts = [
        ("Problem2_灵敏度单独.py", "问题2灵敏度分析"),
        ("Problem3_灵敏度单独.py", "问题3灵敏度分析"), 
        ("Problem4_灵敏度分析.py", "问题4灵敏度分析")
    ]
    
    success_count = 0
    for script, desc in sensitivity_scripts:
        if os.path.exists(script):
            if run_script(script, desc):
                success_count += 1
        else:
            print(f"⚠️  脚本不存在: {script}")
    
    print(f"🔬 灵敏度分析完成: {success_count}/{len(sensitivity_scripts)} 个成功\n")
    return success_count == len(sensitivity_scripts)

def show_results():
    """显示输出结果"""
    print("📁 输出结果文件:")
    print()
    
    if os.path.exists("output"):
        for root, dirs, files in os.walk("output"):
            level = root.replace("output", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # 只显示前5个文件
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... 还有 {len(files) - 5} 个文件")
    else:
        print("   (暂无输出文件)")

def main():
    """主函数"""
    print_banner()
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请解决上述问题后重试")
        return
    
    # 询问用户要运行什么
    print("请选择要运行的内容:")
    print("1. 仅运行主解决方案")
    print("2. 仅运行灵敏度分析") 
    print("3. 运行所有内容")
    print("4. 仅检查输出结果")
    
    try:
        choice = input("请输入选择 (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
        return
    
    print()
    
    success = True
    
    if choice == "1":
        success = run_all_solutions()
    elif choice == "2":
        success = run_sensitivity_analysis()
    elif choice == "3":
        success1 = run_all_solutions()
        success2 = run_sensitivity_analysis()
        success = success1 and success2
    elif choice == "4":
        show_results()
        return
    else:
        print("❌ 无效选择")
        return
    
    # 显示总结
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    
    print("=" * 80)
    if success:
        print("🎉 所有任务运行成功!")
    else:
        print("⚠️  部分任务运行失败，请检查上述错误信息")
    
    print(f"⏰ 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总耗时: {elapsed_time}")
    print()
    
    show_results()
    print()
    print("📊 详细结果请查看 output/ 目录下的相应文件")
    print("=" * 80)

if __name__ == "__main__":
    main()