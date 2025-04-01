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
   cd BayesianOptimizatio
   ```

3. 使用Docker Compose启动应用
   ```bash
   docker-compose up -d
   ```

4. 在浏览器中访问 http://localhost:8501

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
