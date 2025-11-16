# 贡献指南 (Contributing Guide)

感谢您对2025-CUMCM项目的关注！我们欢迎任何形式的贡献，无论是报告问题、提出建议，还是提交代码。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [提交流程](#提交流程)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [文档贡献](#文档贡献)
- [问题报告](#问题报告)
- [功能请求](#功能请求)

## 🤝 行为准则

### 我们的承诺
为了营造一个开放和友好的环境，我们作为贡献者和维护者承诺，无论年龄、体型、残疾、种族、性别认同和表达、经验水平、国籍、个人形象、种族、宗教或性取向如何，参与我们项目和社区的每个人都能享受无骚扰的体验。

### 我们的标准
有助于创造积极环境的行为包括：

- ✅ 使用友好和包容的语言
- ✅ 尊重不同的观点和经验
- ✅ 优雅地接受建设性批评
- ✅ 关注对社区最有利的事情
- ✅ 对其他社区成员表示同理心

不可接受的行为包括：

- ❌ 使用性化的语言或图像，以及不受欢迎的性关注或性骚扰
- ❌ 恶意评论、侮辱/贬损评论，以及个人或政治攻击
- ❌ 公开或私下骚扰
- ❌ 未经明确许可，发布他人的私人信息，如物理或电子地址
- ❌ 在专业环境中可能被合理认为不适当的其他行为

## 🚀 如何贡献

### 1. 报告问题 (Bug Reports)

如果您发现了bug，请通过以下步骤报告：

1. **检查现有Issues**: 确保问题尚未被报告
2. **创建新Issue**: 使用适当的模板
3. **提供详细信息**:
   - 问题描述和重现步骤
   - 期望行为和实际行为
   - 环境信息（操作系统、Python/Java版本等）
   - 相关的错误日志或截图
4. **添加标签**: 选择合适的标签（bug、enhancement等）

### 2. 功能请求 (Feature Requests)

我们欢迎新功能的建议！请：

1. **描述用例**: 详细说明为什么需要这个功能
2. **提出解决方案**: 如果有想法，请描述实现方式
3. **考虑替代方案**: 是否有其他方法可以解决同样的问题
4. **添加上下文**: 说明功能的重要性和影响范围

### 3. 代码贡献

我们欢迎代码贡献！请遵循以下步骤：

#### 步骤1: Fork和Clone

```bash
# Fork项目到您的GitHub账户
# 然后克隆您的fork
git clone https://github.com/YOUR-USERNAME/2025-CUMCM.git
cd 2025-CUMCM

# 添加上游仓库
git remote add upstream https://github.com/Kong-Shuo/2025-CUMCM.git
```

#### 步骤2: 创建分支

```bash
# 确保主分支是最新的
git checkout main
git pull upstream main

# 创建新分支
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

#### 步骤3: 开发和测试

- 编写代码实现您的功能
- 确保遵循代码规范
- 添加必要的测试
- 验证所有测试通过

#### 步骤4: 提交更改

```bash
# 添加更改
git add .

# 提交（使用清晰的提交信息）
git commit -m "feat: add new thickness calculation algorithm"

# 推送到您的fork
git push origin feature/your-feature-name
```

#### 步骤5: 创建Pull Request

1. 访问GitHub上的您的fork
2. 点击"New Pull Request"
3. 选择正确的分支
4. 填写PR模板
5. 等待代码审查

## 🛠 开发环境设置

### 系统要求

- **操作系统**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.7+
- **Java**: 17+
- **Maven**: 3.6+

### Python环境设置

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt
```

### Java环境设置

```bash
# 编译项目
cd Simulation
mvn clean compile

# 运行测试
mvn test

# 启动应用
mvn spring-boot:run
```

### IDE配置

推荐使用以下IDE：

- **Python**: PyCharm, VS Code
- **Java**: IntelliJ IDEA, Eclipse
- **通用**: VS Code

#### VS Code扩展推荐

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "redhat.java",
    "vscjava.vscode-java-pack",
    "ms-vscode.vscode-thunder-client"
  ]
}
```

## 📝 提交流程

### 提交信息规范

我们使用[Conventional Commits](https://www.conventionalcommits.org/)规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### 类型 (Type)

- `feat`: 新功能
- `fix`: bug修复
- `docs`: 文档更新
- `style`: 代码格式化（不影响功能）
- `refactor`: 代码重构
- `test`: 添加或修改测试
- `chore`: 构建过程或辅助工具的变动

#### 示例

```bash
feat(algorithm): add improved thickness calculation method

Implement new algorithm based on multi-beam interference theory
with better accuracy for SiC epitaxial layers.

Closes #123
```

### Pull Request规范

#### PR标题

使用与提交信息相同的格式。

#### PR描述

请包含以下内容：

1. **变更概述**: 简要描述变更内容
2. **动机**: 为什么需要这个变更
3. **测试**: 如何测试这些变更
4. **截图**: 如果是UI变更，请提供截图
5. **相关问题**: 链接相关的Issue

#### PR检查清单

在提交PR前，请确认：

- [ ] 代码遵循项目规范
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] 提交信息清晰明确
- [ ] 没有合并冲突

## 📐 代码规范

### Python代码规范

我们遵循[PEP 8](https://www.python.org/dev/peps/pep-0008/)规范：

#### 命名规范

```python
# 类名：大驼峰命名
class ThicknessCalculator:
    pass

# 函数和变量：小写下划线
def calculate_thickness(wavelength, refractive_index):
    incident_angle = 10.0
    return thickness

# 常量：大写下划线
MAX_THICKNESS = 100.0
MIN_WAVELENGTH = 2500
```

#### 文档字符串

```python
def calculate_reflectance(wavelength, thickness, incident_angle):
    """
    Calculate reflectance using multi-beam interference theory.
    
    Args:
        wavelength (float): Wavelength in nanometers
        thickness (float): Layer thickness in micrometers
        incident_angle (float): Incident angle in degrees
    
    Returns:
        float: Reflectance value (0-1)
    
    Raises:
        ValueError: If parameters are out of valid range
    
    Example:
        >>> r = calculate_reflectance(5000, 7.32, 10.0)
        >>> print(f"Reflectance: {r:.3f}")
    """
    pass
```

#### 类型注解

```python
from typing import List, Tuple, Optional

def detect_peaks(data: np.ndarray, 
                 prominence: float = 0.05) -> Tuple[np.ndarray, dict]:
    """Detect peaks in spectral data."""
    pass
```

### Java代码规范

我们遵循[Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)：

#### 命名规范

```java
// 类名：大驼峰
public class SiCMeasurementService {
    
    // 常量：大写下划线
    private static final double DEFAULT_REFRACTIVE_INDEX = 2.55;
    
    // 变量和方法：小驼峰
    private double incidentAngle;
    
    public double calculateThickness() {
        return thickness;
    }
}
```

#### 注释规范

```java
/**
 * Service for calculating SiC epitaxial layer thickness using
 * multi-beam interference theory.
 * 
 * @author Kong-Shuo
 * @version 1.0
 * @since 2025-01-XX
 */
@Service
public class SiCMeasurementService {
    
    /**
     * Calculates thickness based on interference pattern.
     * 
     * @param request measurement parameters
     * @return calculation result
     * @throws IllegalArgumentException if parameters are invalid
     */
    public MeasurementResult calculateThickness(MeasurementRequest request) {
        // implementation
    }
}
```

## 🧪 测试要求

### 测试覆盖率

- **Python**: 目标覆盖率 > 80%
- **Java**: 目标覆盖率 > 75%

### 测试类型

#### 单元测试

```python
# Python示例
import unittest
from src.algorithm import calculate_thickness

class TestThicknessCalculation(unittest.TestCase):
    
    def test_valid_parameters(self):
        """Test thickness calculation with valid parameters."""
        result = calculate_thickness(5000, 7.32, 10.0)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_thickness(-1000, 7.32, 10.0)
```

```java
// Java示例
@ExtendWith(MockitoExtension.class)
class SiCMeasurementServiceTest {
    
    @InjectMocks
    private SiCMeasurementService service;
    
    @Test
    void calculateThickness_ValidParameters_ReturnsCorrectResult() {
        // Given
        MeasurementRequest request = new MeasurementRequest();
        request.setIncidentAngle(10.0);
        request.setThickness(7.32);
        
        // When
        MeasurementResult result = service.calculateThickness(request);
        
        // Then
        assertThat(result.getThickness()).isGreaterThan(0);
    }
}
```

#### 集成测试

```python
class TestIntegration(unittest.TestCase):
    
    def test_full_calculation_pipeline(self):
        """Test the complete calculation pipeline."""
        # Test data loading, processing, and calculation
        pass
```

### 运行测试

```bash
# Python测试
cd 模型代码
python -m pytest tests/ -v --cov=src

# Java测试
cd Simulation
mvn test
mvn verify  # 包括集成测试
```

## 📚 文档贡献

### 文档类型

1. **API文档**: 函数和类的详细说明
2. **用户指南**: 如何使用项目的说明
3. **开发文档**: 架构和设计决策
4. **示例代码**: 使用示例和教程

### 文档规范

- 使用清晰、简洁的语言
- 提供代码示例
- 包含必要的图表和截图
- 保持文档与代码同步

### 文档生成

```bash
# Python文档
cd docs
make html

# Java文档 (Javadoc)
cd Simulation
mvn javadoc:javadoc
```

## 🐛 问题报告模板

使用以下模板报告问题：

```markdown
## Bug描述
简要描述遇到的问题

## 重现步骤
1. 进入 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

## 期望行为
描述您期望发生的情况

## 实际行为
描述实际发生的情况

## 环境信息
- 操作系统: [e.g. Windows 10, macOS 11.0, Ubuntu 20.04]
- Python版本: [e.g. 3.8.10]
- Java版本: [e.g. 17.0.2]
- 项目版本: [e.g. v1.0.0]

## 错误日志
```
粘贴相关的错误日志
```

## 附加信息
添加任何其他有助于解决问题的信息
```

## 💡 功能请求模板

```markdown
## 功能描述
简要描述您希望添加的功能

## 问题背景
描述这个功能要解决的问题

## 解决方案
描述您希望的解决方案

## 替代方案
描述您考虑过的其他解决方案

## 附加信息
添加任何其他相关信息或截图
```

## 🏆 贡献者认可

我们会在以下地方认可贡献者：

1. **README.md**: 主要贡献者列表
2. **CHANGELOG.md**: 版本更新中的贡献说明
3. **贡献者页面**: 详细的贡献者介绍

### 贡献类型

- 💻 代码贡献
- 📖 文档改进
- 🐛 Bug报告
- 💡 功能建议
- 🎨 设计改进
- 🌍 翻译工作
- 📢 推广宣传

## 📞 联系方式

如果您有任何问题或需要帮助，请通过以下方式联系我们：

- **GitHub Issues**: [提交问题](https://github.com/Kong-Shuo/2025-CUMCM/issues)
- **邮箱**: [your-email@example.com]
- **讨论区**: [GitHub Discussions](https://github.com/Kong-Shuo/2025-CUMCM/discussions)

## 📄 许可证

通过贡献代码，您同意您的贡献将在[MIT License](LICENSE)下授权。

---

感谢您的贡献！🎉