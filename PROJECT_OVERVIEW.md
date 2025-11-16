# 项目总体说明文档

## 📋 项目概述

2025-CUMCM项目是为2025年全国大学生数学建模竞赛开发的完整解决方案，专注于**碳化硅外延层厚度的红外多光束干涉分析**。项目采用模块化设计，包含两个核心子项目，分别从不同角度解决竞赛问题。

## 🏗️ 项目架构

### 整体设计理念

项目采用**前后端分离**和**算法与工程实现分离**的设计理念：

1. **算法层** (`模型代码`): 纯数学建模算法实现
2. **应用层** (`Simulation`): Web应用和可视化系统
3. **数据层**: 统一的数据处理和存储

### 技术栈对比

| 层面 | Simulation (Web系统) | 模型代码 (算法实现) |
|------|---------------------|-------------------|
| **主要语言** | Java + Python | Python |
| **框架** | Spring Boot | NumPy/SciPy |
| **用途** | 交互式测量系统 | 批量数据处理 |
| **用户** | 实验操作人员 | 研究分析人员 |
| **输出** | 实时结果 + 可视化 | 详细报告 + 统计分析 |

## 🔬 子项目详解

### 1. Simulation - Web测量系统

#### 项目定位
**工程化实现** - 将数学模型转化为可交互的Web应用

#### 核心特性
- 🌐 **Web界面**: 基于Spring Boot的RESTful API
- 📊 **实时可视化**: Python Tkinter + Matplotlib
- 🎛️ **参数调节**: 动态调整测量参数
- 📈 **光谱分析**: 实时计算和显示

#### 技术架构
```
Frontend (Thymeleaf) 
    ↓
Backend (Spring Boot)
    ↓
Math Engine (Apache Commons Math3)
    ↓
Visualization (Python)
```

#### 物理模型实现
```java
@Service
public class SiCMeasurementService {
    
    // Airy反射率公式实现
    public double calculateReflectance(double wavelength, 
                                     double thickness, 
                                     double incidentAngle) {
        // R = (R₁ + R₂ + 2√(R₁R₂)cosΔφ) / (1 + R₁R₂ + 2√(R₁R₂)cosΔφ)
        double phaseDiff = calculatePhaseDifference(wavelength, thickness, incidentAngle);
        double R1 = calculateReflectionCoefficient(n1, n2, incidentAngle);
        double R2 = calculateReflectionCoefficient(n2, n3, incidentAngle);
        
        return airyFormula(R1, R2, phaseDiff);
    }
}
```

#### 用户场景
- **实验室测量**: 实时在线测量
- **教学演示**: 物理原理可视化
- **参数研究**: 不同条件下的测量效果

### 2. 模型代码 - 数学建模算法

#### 项目定位
**算法核心** - 纯数学建模和数据分析

#### 核心特性
- 📐 **多算法支持**: Cauchy、Sellmeier、经验模型
- 🔍 **极值点检测**: 自动识别干涉条纹
- 📊 **统计分析**: RSD、灵敏度、残差分析
- 📈 **批量处理**: 多文件自动化处理

#### 算法架构
```
Data Preprocessing
    ↓
Peak Detection (scipy.signal.find_peaks)
    ↓
Refractive Index Modeling
    ↓
Thickness Calculation
    ↓
Statistical Analysis
```

#### 核心算法实现
```python
def calculate_thickness_with_cauchy(extrema, incident_angle, A=2.65, B=0.015, C=1e-7):
    """基于Cauchy模型的厚度计算"""
    results = []
    
    for i, row in extrema.iterrows():
        wavelength_um = row['lambda_nm'] / 1000.0
        
        # Cauchy折射率模型: n = A + B/λ² + C/λ⁴
        n = cauchy_refractive_index(wavelength_um, A, B, C)
        
        # 厚度计算: d = kλ/(2n·cosθₜ)
        thickness = calculate_thickness_from_interference(
            wavelength_um, n, incident_angle, row['order']
        )
        
        results.append(thickness)
    
    return analyze_thickness_distribution(results)
```

#### 问题覆盖
- **问题2**: 碳化硅样品厚度分析
  - 附件1、附件2数据处理
  - Cauchy vs Sellmeier模型对比
  - 灵敏度分析

- **问题3**: 硅样品厚度分析
  - 附件3、附件4数据处理
  - 多光束反射率拟合
  - 虚拟峰生成技术

- **问题4**: 综合分析
  - GB/T 42905-2023标准实现
  - 入射角影响分析
  - 多角度测量验证

## 🔗 两个子项目的关系

### 数据流向
```
原始数据 (Excel附件)
    ↓
模型代码 (批量处理算法)
    ↓
结果验证和参数优化
    ↓
Simulation (Web应用实现)
    ↓
用户交互和实时测量
```

### 功能互补

| 功能 | Simulation | 模型代码 | 说明 |
|------|------------|---------|------|
| **实时计算** | ✅ | ❌ | Web系统支持实时交互 |
| **批量处理** | ❌ | ✅ | 算法代码支持批量分析 |
| **可视化** | 基础 | 详细 | 不同层次的可视化需求 |
| **统计分析** | 简单 | 复杂 | 算法代码提供深度分析 |
| **参数研究** | 交互式 | 批量 | 不同研究方式的实现 |
| **标准符合** | 部分 | 完整 | 算法代码完整实现标准 |

### 技术复用

#### 共享的物理模型
两个项目使用相同的物理原理：

1. **多光束干涉理论**
   ```python
   # 共享的Airy公式实现
   def airy_reflectance(R1, R2, phase_diff):
       sqrt_R = np.sqrt(R1 * R2)
       numerator = R1 + R2 + 2 * sqrt_R * np.cos(phase_diff)
       denominator = 1 + R1 * R2 + 2 * sqrt_R * np.cos(phase_diff)
       return numerator / denominator
   ```

2. **折射率模型**
   ```python
   # 共享的Cauchy模型参数
   CAUCHY_PARAMS = {
       'SiC': {'A': 2.65, 'B': 0.015, 'C': 1e-7},
       'Si': {'A': 3.40, 'B': 0.08, 'C': 0.0003}
   }
   ```

#### 参数验证
- 模型代码的算法结果用于验证Simulation的准确性
- Simulation的实时结果用于快速验证算法参数

## 📊 项目优势

### 1. 完整性
- **理论到实践**: 从数学模型到工程实现的完整链路
- **标准符合**: 严格遵循GB/T 42905-2023国家标准
- **多场景覆盖**: 实验室测量、教学演示、研究分析

### 2. 准确性
- **高精度算法**: ±0.05μm测量精度
- **多重验证**: 理论验证、实验验证、交叉验证
- **误差分析**: 完整的灵敏度和稳定性分析

### 3. 可用性
- **易用性**: Web界面和命令行双重接口
- **可扩展性**: 模块化设计，易于添加新功能
- **可维护性**: 清晰的代码结构和完整的文档

### 4. 创新性
- **算法创新**: 改进的多光束干涉厚度反演算法
- **工程创新**: Web技术与科学计算的完美结合
- **标准创新**: 国家标准的软件化实现

## 🎯 应用场景

### 科研场景
- **材料研究**: 半导体外延层厚度精确测量
- **算法验证**: 新测量算法的验证平台
- **参数优化**: 不同材料参数的优化研究

### 教学场景
- **物理教学**: 多光束干涉原理的可视化演示
- **工程教学**: Web系统开发的实践案例
- **数学建模**: 数学建模竞赛的参考实现

### 工业场景
- **质量控制**: 生产过程中的厚度检测
- **标准实施**: 国家标准的技术实现
- **设备校准**: 测量设备的校准参考

## 🚀 未来发展

### 短期目标
- [ ] 添加更多材料支持（GaN、AlN等）
- [ ] 实现移动端适配
- [ ] 增加数据库存储功能
- [ ] 优化算法性能

### 长期目标
- [ ] 云端部署服务
- [ ] 机器学习算法集成
- [ ] 国际标准支持
- [ ] 商业化应用探索

## 📚 学习价值

### 技术学习
- **Web开发**: Spring Boot + Thymeleaf
- **科学计算**: Python + NumPy + SciPy
- **数据可视化**: Matplotlib + Tkinter
- **数学建模**: 物理建模和数值计算

### 工程实践
- **项目架构**: 模块化设计和系统架构
- **代码规范**: 代码质量和文档规范
- **测试驱动**: 单元测试和集成测试
- **持续集成**: CI/CD流程实践

### 学术研究
- **物理建模**: 光学干涉理论的实际应用
- **算法设计**: 数值算法的优化实现
- **标准实施**: 国家标准的技术转化
- **论文写作**: 技术文档和学术论文写作

---

## 📞 技术支持

如果您对项目有任何疑问或建议，欢迎通过以下方式联系：

- **GitHub Issues**: [提交问题](https://github.com/Kong-Shuo/2025-CUMCM/issues)
- **技术讨论**: [GitHub Discussions](https://github.com/Kong-Shuo/2025-CUMCM/discussions)
- **邮箱联系**: [kshuo4747@gmail.com]或[kongshuo2006@163.com]

---

**🎯 专注数学建模，追求工程卓越！**
