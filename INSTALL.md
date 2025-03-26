# 贝叶斯优化材料组分系统 - 构建指南

本文档提供了在Windows环境下构建和打包贝叶斯优化材料组分系统的详细说明。

## 环境要求

- Windows 10/11 64位操作系统
- Python 3.9或更高版本
- 至少4GB内存
- 至少2GB可用磁盘空间（用于构建环境和生成安装包）

## 构建前准备

### 1. 安装Python

1. 从[Python官网](https://www.python.org/downloads/)下载并安装Python 3.9或更高版本
2. 安装时勾选「Add Python to PATH」选项
3. 安装完成后，打开命令提示符或PowerShell，输入以下命令验证安装：
   ```
   python --version
   ```

### 2. 克隆代码仓库

```
git clone https://github.com/your-username/BO-docker.git
cd BO-docker
```

## 构建方法

### 方法一：使用构建脚本（推荐）

项目提供了自动构建脚本，可以一键完成环境配置、依赖安装和应用打包：

1. 双击项目根目录下的`build.bat`文件
2. 脚本将自动执行以下操作：
   - 创建并激活Python虚拟环境
   - 安装所需依赖项
   - 构建可执行文件
   - 创建Windows安装包（.msi文件）
3. 构建完成后，可执行文件将位于`build\exe.win-amd64-3.9`目录下
4. 安装包将位于`dist`目录下

### 方法二：手动构建

如果自动构建脚本无法正常工作，可以按照以下步骤手动构建：

1. 创建并激活虚拟环境：
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. 安装依赖项：
   ```
   pip install -r requirements.txt
   pip install cx_Freeze
   ```

3. 创建数据目录（如果不存在）：
   ```
   mkdir data
   ```

4. 构建应用：
   ```
   python setup.py build
   ```

5. 创建安装包（可选）：
   ```
   python setup.py bdist_msi
   ```

6. 退出虚拟环境：
   ```
   deactivate
   ```

## 构建输出

成功构建后，将生成以下文件：

- **可执行文件**：位于`build\exe.win-amd64-3.9`目录下的`贝叶斯优化材料组分系统.exe`
- **安装包**：位于`dist`目录下的`.msi`文件

## 安装方法

### 使用安装包安装

1. 双击`dist`目录下的`.msi`文件
2. 按照安装向导进行安装
3. 安装完成后，可以从开始菜单或桌面快捷方式启动应用

### 便携版使用

1. 将`build\exe.win-amd64-3.9`目录下的所有文件复制到任意位置
2. 双击`贝叶斯优化材料组分系统.exe`文件启动应用

## 故障排除

### 构建过程中的常见问题

1. **Python未找到**
   - 确保Python已正确安装并添加到PATH环境变量
   - 尝试在命令行中运行`python --version`验证安装

2. **依赖项安装失败**
   - 确保网络连接正常
   - 尝试使用国内镜像源：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

3. **构建失败**
   - 检查是否有足够的磁盘空间
   - 确保所有依赖项已正确安装
   - 检查`setup.py`文件中的配置是否正确

### 运行时的常见问题

1. **应用无法启动**
   - 确保所有依赖文件都已复制到正确位置
   - 检查是否有杀毒软件阻止应用运行

2. **功能无法正常使用**
   - 确保`data`目录存在并有写入权限
   - 检查日志文件（如果存在）获取错误信息

## 开发环境设置

如果您希望在开发环境中运行和修改应用，可以按照以下步骤设置：

1. 创建并激活虚拟环境：
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. 安装依赖项：
   ```
   pip install -r requirements.txt
   ```

3. 运行应用：
   ```
   streamlit run app/main.py
   ```
   或使用提供的批处理文件：
   ```
   start.bat
   ```

## 注意事项

- 构建过程需要联网下载依赖项
- 构建时间取决于计算机性能和网络速度，通常需要5-15分钟
- 生成的可执行文件较大（约200-300MB），请确保有足够的磁盘空间
- 首次启动应用可能需要较长时间，请耐心等待