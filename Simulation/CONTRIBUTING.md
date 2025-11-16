# 贡献指南

感谢您对碳化硅外延层红外多光束干涉测量系统的关注！我们欢迎所有形式的贡献，包括但不限于：

- 🐛 报告Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- ✨ 开发新功能
- 🧪 编写测试用例

## 🚀 开始贡献

### 环境准备

1. **Fork 项目**
   - 点击右上角的 "Fork" 按钮
   - 将项目克隆到本地：
   ```bash
   git clone https://github.com/Kong-Shuo/2025-CUMCM.git
   cd 2025-CUMCM
   ```

2. **设置开发环境**
   ```bash
   # 安装Java 17+
   # 安装Maven 3.6+
   # 安装Python 3.8+ (可选，用于可视化程序)
   
   # 构建项目
   mvn clean compile
   
   # 运行测试
   mvn test
   ```

3. **配置开发工具**
   - 推荐使用 IntelliJ IDEA 或 Eclipse
   - 安装必要的插件：Lombok, Spring Boot
   - 配置代码格式化规则

### 开发流程

1. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或者
   git checkout -b fix/your-bug-fix
   ```

2. **进行开发**
   - 遵循项目的代码规范
   - 添加必要的注释和文档
   - 确保所有测试通过
   - 为新功能编写测试用例

3. **提交代码**
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   ```

4. **推送并创建PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   - 在GitHub上创建Pull Request
   - 填写详细的PR描述
   - 等待代码审查

## 📝 代码规范

### Java代码规范

1. **命名规范**
   - 类名：使用PascalCase（大驼峰）
   - 方法名和变量名：使用camelCase（小驼峰）
   - 常量：使用UPPER_SNAKE_CASE
   - 包名：使用小写字母

2. **代码风格**
   - 使用4个空格缩进
   - 每行不超过120个字符
   - 方法长度不超过50行
   - 类长度不超过500行

3. **注释规范**
   ```java
   /**
    * 方法功能描述
    * 
    * @param param1 参数1描述
    * @param param2 参数2描述
    * @return 返回值描述
    * @throws Exception 异常描述
    */
   public String methodName(String param1, int param2) throws Exception {
       // 实现代码
   }
   ```

### Python代码规范

1. **遵循PEP 8规范**
2. **使用类型提示**
3. **添加docstring**
   ```python
   def calculate_thickness(wavelength: float, thickness: float) -> float:
       """
       计算碳化硅外延层厚度
       
       Args:
           wavelength: 入射光波长 (μm)
           thickness: 外延层厚度 (μm)
           
       Returns:
           计算得到的厚度值
       """
       # 实现代码
   ```

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
mvn test

# 运行特定测试类
mvn test -Dtest=SiCMeasurementServiceTest

# 生成测试报告
mvn surefire-report:report
```

### 编写测试

1. **单元测试**
   ```java
   @ExtendWith(MockitoExtension.class)
   class SiCMeasurementServiceTest {
       
       @Mock
       private MeasurementRepository repository;
       
       @InjectMocks
       private SiCMeasurementServiceImpl service;
       
       @Test
       void testPerformMeasurement() {
           // 测试代码
       }
   }
   ```

2. **集成测试**
   ```java
   @SpringBootTest
   @AutoConfigureTestDatabase
   class SiCMeasurementControllerTest {
       
       @Autowired
       private TestRestTemplate restTemplate;
       
       @Test
       void testMeasureEndpoint() {
           // 集成测试代码
       }
   }
   ```

## 🐛 Bug报告

报告Bug时，请包含以下信息：

1. **Bug描述**
   - 清晰描述遇到的问题
   - 预期行为 vs 实际行为

2. **复现步骤**
   - 详细的重现步骤
   - 相关的输入参数

3. **环境信息**
   - 操作系统
   - Java版本
   - 浏览器版本
   - 项目版本

4. **错误日志**
   - 完整的错误堆栈
   - 相关的日志输出

5. **截图/录屏**
   - 如果是UI问题，提供截图或录屏

## 💡 功能建议

提出新功能建议时：

1. **功能描述**
   - 清晰描述功能需求
   - 说明使用场景

2. **实现建议**
   - 如果有实现思路，请说明
   - 相关的技术方案

3. **优先级**
   - 说明功能的紧急程度
   - 对项目的重要性

## 📚 文档贡献

文档是项目的重要组成部分，我们欢迎：

1. **改进README**
   - 修正错误信息
   - 添加使用示例
   - 改进说明文档

2. **API文档**
   - 完善接口文档
   - 添加示例代码
   - 说明参数含义

3. **代码注释**
   - 为复杂逻辑添加注释
   - 完善方法文档
   - 添加算法说明

## 🔍 代码审查

### 审查标准

1. **功能正确性**
   - 代码是否实现了预期功能
   - 是否有潜在的bug

2. **代码质量**
   - 代码结构是否清晰
   - 是否遵循编码规范
   - 是否有重复代码

3. **性能考虑**
   - 是否有性能问题
   - 是否有内存泄漏风险

4. **安全性**
   - 是否有安全漏洞
   - 输入验证是否充分

### 审查流程

1. **自动检查**
   - 代码格式检查
   - 静态代码分析
   - 测试覆盖率

2. **人工审查**
   - 功能逻辑审查
   - 代码质量审查
   - 架构设计审查

## 🏷️ 提交信息规范

使用[Conventional Commits](https://www.conventionalcommits.org/)规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 类型说明

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 示例

```bash
feat(measurement): 添加多角度测量功能

- 实现多角度厚度测量算法
- 添加参数验证逻辑
- 更新API文档

Closes #123
```

## 🎉 贡献者认可

所有贡献者都会在项目中得到认可：

1. **贡献者列表**
   - 在README中添加贡献者信息
   - 使用GitHub的Contributors功能

2. **发布说明**
   - 在版本发布时感谢贡献者
   - 记录重要贡献

## 📞 联系方式

如有任何问题，请通过以下方式联系：

- 📧 邮箱: [your-email@example.com]
- 💬 GitHub Issues: [项目Issues页面]
- 📱 微信群: [群二维码或群号]

## 📄 许可证

通过贡献代码，您同意您的贡献将在[MIT License](LICENSE)下发布。

---

再次感谢您的贡献！🙏