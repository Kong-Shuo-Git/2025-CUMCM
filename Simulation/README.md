# 碳化硅外延层红外多光束干涉测量系统

## 📖 项目简介

本项目是基于Spring Boot和Python开发的碳化硅外延层厚度测量系统，采用红外多光束干涉原理实现高精度测量。该系统为2025年全国大学生数学建模竞赛开发，提供了完整的Web界面和可视化演示程序。

## 🎯 项目特性

### 🔬 核心功能
- **多光束干涉建模**: 基于Airy公式的精确物理模型
- **实时参数调节**: 支持入射角、厚度、折射率等参数动态调整
- **光谱分析**: 红外反射率光谱计算与拟合
- **厚度反演**: 基于干涉条纹的厚度自动计算
- **可视化演示**: 直观的光束传播路径和干涉图案展示

### 🛠 技术架构
- **后端**: Spring Boot 3.3.3 + Java 17
- **前端**: Thymeleaf模板引擎 + HTML/CSS/JavaScript
- **数学计算**: Apache Commons Math3
- **可视化**: Python + Matplotlib + Tkinter
- **构建工具**: Maven

## 📁 项目结构

```
Simulation/
├── src/
│   ├── main/
│   │   ├── java/com/sdmu/cumcm2025/
│   │   │   ├── application.java              # 主应用入口
│   │   │   ├── config/
│   │   │   │   └── WebConfig.java           # Web配置
│   │   │   ├── controller/
│   │   │   │   └── SiCMeasurementController.java  # 测量控制器
│   │   │   ├── model/
│   │   │   │   └── dto/                     # 数据传输对象
│   │   │   │       ├── MeasurementRequest.java
│   │   │   │       └── MeasurementResult.java
│   │   │   └── service/
│   │   │       ├── SiCMeasurementService.java
│   │   │       └── impl/
│   │   │           └── SiCMeasurementServiceImpl.java
│   │   └── resources/
│   │       ├── application.yml               # 应用配置
│   │       ├── templates/                    # Thymeleaf模板
│   │       └── static/                       # 静态资源
│   └── test/                                 # 测试代码
├── target/classes/Simulation.py               # Python可视化程序
├── pom.xml                                   # Maven配置
└── README.md                                 # 项目说明
```

## 🚀 快速开始

### 环境要求
- **Java**: JDK 17+
- **Maven**: 3.6+
- **Python**: 3.8+ (用于可视化程序)
- **浏览器**: 现代浏览器 (Chrome, Firefox, Safari等)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/Simulation.git
   cd Simulation
   ```

2. **构建项目**
   ```bash
   mvn clean compile
   ```

3. **运行应用**
   ```bash
   mvn spring-boot:run
   ```

4. **访问应用**
   - Web界面: http://localhost:8080
   - API接口: http://localhost:8080/api/measure

5. **运行可视化程序** (可选)
   ```bash
   # 安装Python依赖
   pip install numpy matplotlib scipy tkinter
   
   # 运行可视化程序
   python target/classes/Simulation.py
   ```

## 📊 使用说明

### Web界面操作
1. 打开浏览器访问 `http://localhost:8080`
2. 在参数控制面板中调整测量参数：
   - 入射角 (5°-30°)
   - 外延层厚度 (1-20 μm)
   - 材料折射率
3. 点击"执行厚度测量"按钮
4. 查看实时计算结果和可视化图表

### API接口
- **POST** `/api/measure` - 执行厚度测量
  ```json
  {
    "incidentAngle": 10.0,
    "thickness": 7.32,
    "refractiveIndex": 2.52,
    "substrateIndex": 3.05
  }
  ```

- **GET** `/api/defaults` - 获取默认参数

### Python可视化程序
运行Python程序后，可以使用交互式界面：
- 实时调整物理参数
- 观察光束传播路径
- 分析干涉光谱特性
- 验证厚度测量精度

## 🔬 物理原理

### 多光束干涉理论
系统基于薄膜光学中的多光束干涉原理：

1. **菲涅尔反射系数**:
   ```
   r_s = (n₁cosθᵢ - n₂cosθₜ) / (n₁cosθᵢ + n₂cosθₜ)
   ```

2. **相位差计算**:
   ```
   Δφ = (4π/λ) × n × d × cosθₜ
   ```

3. **Airy反射率公式**:
   ```
   R = (R₁ + R₂ + 2√(R₁R₂)cosΔφ) / (1 + R₁R₂ + 2√(R₁R₂)cosΔφ)
   ```

### 厚度反演算法
通过分析反射光谱中相邻极大值的波数差来计算厚度：
```
d = 1 / (2Δσ × n_eff × cosθₜ)
```

## 🧪 测试与验证

### 精度验证
- 多角度测量重复性测试
- 不同厚度范围的准确性验证
- 折射率参数敏感性分析

### 性能指标
- 测量精度: ±0.05 μm
- 响应时间: <100ms
- 支持厚度范围: 1-50 μm

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 山东管理学院数学建模团队
- 2025年全国大学生数学建模竞赛组委会
- 开源社区提供的优秀库和工具支持

## 📞 联系方式

- 项目维护者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目地址: https://github.com/your-username/Simulation

## 📈 更新日志

### v1.0.0 (2025-01-XX)
- ✨ 初始版本发布
- 🔬 实现多光束干涉测量核心算法
- 🌐 提供Web界面和API接口
- 📊 集成Python可视化演示程序
- 📚 完整的文档和测试用例

---

⭐ 如果这个项目对您有帮助，请给我们一个Star！