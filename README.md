# 贝叶斯优化材料组分系统

这是一个基于Docker的贝叶斯优化系统，用于材料组分、工艺条件等的优化。该系统允许用户通过迭代实验和机器学习模型来优化材料性能，无需在本地安装Python环境或其他库。

## 功能特点

1. **数据上传与处理**：
   - 支持从本地上传CSV文件
   - 自动转换为数据表格式
   - 可选择特定列作为特征和优化目标

2. **多模型训练与评估**：
   - 支持多种机器学习模型：Lasso、随机森林、XGBoost、SVR和高斯过程
   - 自动进行交叉验证(CV)和超参数优化
   - 比较测试集上的性能指标(R²或RMSE)
   - 自动选择最优模型并使用全部数据进行训练

3. **特征空间搜索与优化**：
   - 可定义特征的范围和步长
   - 可定义特征的范围和步长，用于构建搜索空间
   - 支持多种特征空间搜索策略：网格搜索 (Grid Search)、随机搜索 (Random Search)、遗传算法 (Genetic Algorithm, GA)、粒子群优化 (Particle Swarm Optimization, PSO)、模拟退火 (Simulated Annealing, SA)
   - 使用最优模型和选定的搜索策略进行优化搜索
   - 支持多种采集函数 (Acquisition Functions) 如：期望提升 (Expected Improvement, EI)、置信上界 (Upper Confidence Bound, UCB)、改进概率 (Probability of Improvement, PI) 等，用于推荐下一轮实验条件
   - 可自定义推荐的实验数量
   - **新增特征和约束功能**：支持限制所有特征值的总和不超过指定值，适用于材料组分优化等场景

4. **迭代优化**：
   - 支持补充新的实验数据
   - 重新训练模型并更新推荐
   - 循环迭代，不断优化实验条件

## 快速开始

### 使用Docker（推荐）

1. 确保已安装 [Docker](https://www.docker.com/products/docker-desktop/) 和 [Docker Compose](https://docs.docker.com/compose/install/)

2. 克隆本仓库
   ```bash
   git clone https://github.com/E2zzreal/BayesianOptimization.git
   cd BayesianOptimization
   ```

3. 使用Docker Compose启动应用
   ```bash
   docker-compose up -d
   ```

4. 在浏览器中访问 http://localhost:8501

### 从Dockerfile构建并运行（可选）

如果您不想使用Docker Compose，或者想要自定义构建过程，可以手动构建Docker镜像并运行容器：

1.  **构建镜像**：在项目根目录下运行以下命令来构建Docker镜像。将 `your-image-name` 替换为您想要的镜像名称（例如 `bo-app`）。
    ```bash
    docker build -t your-image-name .
    ```

2.  **运行容器**：使用构建好的镜像启动一个容器。
    ```bash
    docker run -p 8501:8501 your-image-name
    ```
    *   `-p 8501:8501`：将主机的8501端口映射到容器的8501端口。
    *   `your-image-name`：您在构建步骤中指定的镜像名称。

3.  **访问应用**：在浏览器中访问 http://localhost:8501

### 打包镜像并在其他机器运行

如果您需要将构建好的镜像迁移到另一台没有网络连接或无法直接访问 Docker Hub 的机器上运行，可以按以下步骤操作：

1.  **保存镜像为文件**：在构建镜像的机器上，使用 `docker save` 命令将镜像导出为 `.tar` 文件。
    ```bash
    # 将 your-image-name 替换为您构建时使用的镜像名
    docker save -o your-image-name.tar your-image-name
    ```

2.  **传输文件**：将生成的 `your-image-name.tar` 文件复制到目标机器。

3.  **加载镜像**：在目标机器上，使用 `docker load` 命令从 `.tar` 文件加载镜像。
    ```bash
    docker load -i your-image-name.tar
    ```
    加载后，可以使用 `docker images` 确认镜像已存在。

4.  **运行容器**：使用加载的镜像运行容器。
    ```bash
    docker run -p 8501:8501 your-image-name
    ```

5.  **访问应用**：在目标机器的浏览器中访问 `http://localhost:8501`。

### 本地运行

1. 确保已安装Python 3.9或更高版本

2. 克隆本仓库
   ```bash
   git clone https://github.com/your-username/BO-docker.git
   cd BO-docker
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

4. 运行应用
   ```bash
   streamlit run app/main.py
   ```
   或使用提供的批处理文件
   ```bash
   start.bat
   ```

## 使用指南

1. **数据上传**：
   - 点击"上传CSV文件"按钮
   - 选择包含实验数据的CSV文件
   - 选择作为特征的列和目标列
   - 系统会自动初始化特征搜索空间

2. **模型训练**：
   - 点击"训练模型"按钮
   - 系统将自动训练多个模型并显示评估结果
   - 选择最佳模型用于后续优化
   - 设置评估指标（R²或RMSE）
   - 设置训练测试集划分比例和交叉验证折数

3. **特征空间定义与优化设置**：
   - 为每个特征设置范围和步长
   - 为每个特征设置合理的范围和步长
   - 点击"更新特征空间"按钮（系统会估算空间大小和潜在的内存消耗）
   - 选择优化搜索策略（例如：网格搜索、遗传算法、粒子群优化、模拟退火、随机搜索）
   - 选择采集函数（例如：EI、UCB、PI）
   - 设置优化目标（最大化或最小化）
   - （可选，主要用于非高斯过程/随机森林模型）设置Bootstrap模型数量以估计预测不确定性
   - **新增约束设置**：在"高级约束设置"中可启用特征和约束，限制所有特征值的总和不超过指定值

4. **实验推荐**：
   - （确保已完成特征空间定义与优化设置）
   - 设置需要推荐的实验数量
   - 点击"生成实验推荐"按钮，系统将结合最优模型、选定的搜索策略和采集函数，计算并推荐下一轮最有潜力的实验条件
   - 查看生成的实验推荐，并可下载为CSV文件

5. **迭代优化**：
   - 完成推荐实验后，上传新的实验结果
   - 点击"更新模型"按钮重新训练
   - 获取新的实验推荐，继续迭代
   - 查看优化历史和进程图

## 项目结构

```
├── app/                    # 应用主目录
│   ├── main.py             # 主应用入口
│   ├── models/             # 机器学习模型
│   ├── optimization/       # 贝叶斯优化相关代码
│   └── utils/              # 工具函数
├── data/                   # 数据存储目录
├── Dockerfile              # Docker配置文件
├── docker-compose.yml      # Docker Compose配置
├── requirements.txt        # 项目依赖
├── start.bat               # Windows启动脚本
└── README.md               # 项目说明
```

## 依赖项

- Python 3.9+
- scikit-learn 1.2.0+
- xgboost 1.7.0+
- streamlit 1.13.0+
- scikit-optimize 0.9.0+
- GPy 1.10.0+
- GPyOpt 1.2.6+
- joblib 1.2.0+
- pandas 1.5.0+
- numpy 1.23.0+

## 技术栈

- **前端**：Streamlit
- **后端**：Python
- **机器学习**：scikit-learn, XGBoost, GPy
- **优化算法**：scikit-optimize, GPyOpt
- **容器化**：Docker

## 贡献指南

欢迎提交问题和拉取请求，共同改进这个项目。

## 许可证

[MIT](LICENSE)
