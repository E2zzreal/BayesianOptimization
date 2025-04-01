# 导入必要的库
import streamlit as st  # Streamlit用于构建Web应用界面
import pandas as pd     # Pandas用于数据处理和分析
import numpy as np      # NumPy用于数值计算
import os               # 操作系统接口
import sys              # 系统相关功能
import logging          # 日志记录
from pathlib import Path  # 路径操作工具
from datetime import datetime  # 日期时间处理

# 添加项目根目录到系统路径，以便能够导入自定义模块
sys.path.append(str(Path(__file__).parent.parent))

# 创建日志目录（如果不存在）
app_dir = str(Path(__file__).parent.parent)
logs_dir = os.path.join(app_dir, "logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 设置日志记录
def setup_logging():
    """配置日志系统"""
    # 创建日志文件名（包含日期）
    log_filename = os.path.join(logs_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

    # 配置日志格式
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(log_format)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有处理器（避免重复）
    if logger.handlers:
        logger.handlers.clear()

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 初始化日志系统
logger = setup_logging()

# 检查是否已经记录过启动信息
if 'app_started_logged' not in st.session_state:
    logger.info("贝叶斯优化材料组分系统启动中...")
    # 标记为已记录
    st.session_state.app_started_logged = True

# 导入自定义模块
from app.utils.data_processor import DataProcessor  # 数据处理工具类
from app.models.model_trainer import ModelTrainer    # 模型训练和评估类
from app.optimization.bayesian_optimizer import BayesianOptimizer  # 贝叶斯优化器类

# 设置页面配置
st.set_page_config(
    page_title="贝叶斯优化材料组分系统",  # 页面标题
    page_icon="🧪",                     # 页面图标
    layout="wide",                      # 使用宽布局
    initial_sidebar_state="expanded"    # 初始侧边栏状态为展开
)

# 初始化会话状态，用于在不同页面间共享数据
if 'data' not in st.session_state:
    st.session_state.data = None  # 存储上传的数据
if 'features' not in st.session_state:
    st.session_state.features = None  # 存储选择的特征列
if 'target' not in st.session_state:
    st.session_state.target = None  # 存储选择的目标列
if 'model' not in st.session_state:
    st.session_state.model = None  # 存储训练好的最佳模型
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None  # 存储最佳模型名称
if 'feature_ranges' not in st.session_state:
    st.session_state.feature_ranges = {}  # 存储特征的范围信息
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None  # 存储优化推荐结果
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []  # 存储优化历史记录
if 'n_bootstraps' not in st.session_state:
    st.session_state.n_bootstraps = 50  # 默认Bootstrap模型数量，用于非GP和非随机森林模型的不确定性估计
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor() # 初始化数据处理器

# 主页面标题
st.title("贝叶斯优化材料组分系统")

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择功能页面",
    ["数据上传与处理", "模型训练与评估", "特征输入与预测", "搜索推荐", "迭代优化"] # 更新页面列表
)

# 数据上传与处理页面
if page == "数据上传与处理":
    """
    数据上传与处理页面功能：
    1. 允许用户上传CSV格式的数据文件
    2. 预览上传的数据
    3. 选择特征列和目标列
    4. 初始化特征范围信息
    """
    st.header("数据上传与处理")

    # 文件上传组件，限制为CSV格式
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])

    if uploaded_file is not None:
        try:
            # 读取上传的CSV文件数据
            data = pd.read_csv(uploaded_file, index_col=0)
            # 将数据存入会话状态，供其他页面使用
            st.session_state.data = data

            # 显示数据预览
            st.subheader("数据预览")
            # 显示数据前5行，让用户确认数据格式正确
            st.dataframe(data.head())

            # 选择特征和目标列
            st.subheader("选择特征和目标列")
            # 获取数据所有列名
            all_columns = data.columns.tolist()

            # 选择特征列
            features = st.multiselect(
                "选择特征列",
                all_columns,
                default=all_columns[:-1]  # 默认选择除最后一列外的所有列
            )

            # 选择目标列
            remaining_columns = [col for col in all_columns if col not in features]
            target = st.selectbox("选择目标列", remaining_columns)

            if st.button("确认选择"):
                if len(features) > 0 and target:
                    st.session_state.features = features
                    st.session_state.target = target

                    # 初始化特征范围
                    feature_ranges = {}
                    for feature in features:
                        min_val = float(data[feature].min())
                        max_val = float(data[feature].max())
                        feature_ranges[feature] = {
                            "min": min_val,
                            "max": max_val,
                            "step": (max_val - min_val) / 10  # 默认步长
                        }
                    st.session_state.feature_ranges = feature_ranges

                    st.success(f"已选择 {len(features)} 个特征列和 '{target}' 作为目标列")
                else:
                    st.error("请至少选择一个特征列和一个目标列")

        except Exception as e:
            error_msg = f"处理数据时出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)

# 模型训练与评估页面
elif page == "模型训练与评估":
    """
    模型训练与评估页面功能：
    1. 选择要训练的模型类型
    2. 设置评估指标和训练参数
    3. 训练并评估模型性能
    4. 选择最佳模型用于后续优化
    """
    st.header("模型训练与评估")

    # 检查是否已上传数据并选择特征和目标列
    if st.session_state.data is None or st.session_state.features is None or st.session_state.target is None:
        st.warning("请先上传数据并选择特征和目标列")
    else:
        st.info(f"当前数据集: {st.session_state.data.shape[0]} 行, {st.session_state.data.shape[1]} 列")
        st.info(f"选择的特征: {', '.join(st.session_state.features)}")
        st.info(f"选择的目标: {st.session_state.target}")

        # 模型选择
        st.subheader("模型选择")
        # 提供多种回归模型选项，用户可多选
        models_to_train = st.multiselect(
            "选择要训练的模型",
            ["Lasso", "随机森林", "XGBoost", "SVR", "高斯过程"],
            default=["Lasso", "随机森林", "XGBoost", "SVR", "高斯过程"]
        )

        # 评估指标选择
        metric = st.radio("选择评估指标", ["R²", "RMSE"])

        # 训练测试集划分比例
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)

        # 交叉验证折数
        cv_folds = st.slider("交叉验证折数", 3, 10, 5, 1)

        if st.button("训练模型"):
            with st.spinner("正在训练模型，请稍候..."):
                try:
                    # 准备数据
                    X_original = st.session_state.data[st.session_state.features]
                    y = st.session_state.data[st.session_state.target]

                    # 应用特征缩放
                    logger.info("应用特征缩放...")
                    X_scaled = st.session_state.data_processor.scale_features(X_original.copy(), st.session_state.features) # 使用副本避免修改原始数据
                    logger.info(f"特征缩放完成。Scaler: {st.session_state.data_processor.scaler}")

                    # 初始化模型训练器
                    trainer = ModelTrainer(
                        models=models_to_train,
                        metric=metric.lower(),
                        cv_folds=cv_folds,
                        test_size=test_size
                    )

                    # 使用缩放后的数据训练模型
                    logger.info("开始模型训练...")
                    results = trainer.train_and_evaluate(X_scaled, y)
                    logger.info("模型训练完成。")

                    # 显示结果
                    st.subheader("模型评估结果")

                    # 创建结果数据框
                    results_df = pd.DataFrame(results).sort_values(
                        by="cv_score",
                        ascending=True if metric.lower() == "rmse" else False
                    )

                    # 格式化结果
                    results_df["test_score"] = results_df["test_score"].apply(lambda x: f"{x:.4f}")
                    results_df["cv_score"] = results_df["cv_score"].apply(lambda x: f"{x:.4f}")
                    results_df["train_score"] = results_df["train_score"].apply(lambda x: f"{x:.4f}")

                    # 显示结果表格
                    display_df = pd.DataFrame([{
                        '模型名称': r['model_name'],
                        '测试分数': r['test_score'],
                        '交叉验证分数': r['cv_score'],
                        '训练分数': r['train_score']
                    } for r in results])

                    st.dataframe(display_df)

                    # 获取最佳模型
                    best_model_name = results_df.iloc[0]["model_name"]
                    best_model = trainer.get_best_model()

                    # 保存到会话状态
                    st.session_state.model = best_model
                    st.session_state.best_model_name = best_model_name

                    st.success(f"模型训练完成！最佳模型: {best_model_name}")
                    st.info(f"特征缩放器 (Scaler) 已拟合并存储。")

                except Exception as e:
                    error_msg = f"模型训练时出错: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)

# 特征输入与预测页面 (新页面)
elif page == "特征输入与预测":
    st.header("特征输入与预测")

    if st.session_state.model is None:
        st.warning("请先训练模型")
    elif st.session_state.features is None:
        st.warning("请先在'数据上传与处理'页面选择特征")
    else:
        st.info(f"当前最佳模型: {st.session_state.best_model_name}")
        st.markdown("在下方输入特征值，获取预测结果")

        # 创建多列布局用于输入特征值
        feature_values = {}
        cols = st.columns(3)

        for i, feature in enumerate(st.session_state.features):
            col_idx = i % 3
            with cols[col_idx]:
                # 获取当前范围作为参考
                current_range = st.session_state.feature_ranges.get(feature, {})
                min_val = current_range.get("min", 0.0)
                max_val = current_range.get("max", 1.0)
                default_val = float((min_val + max_val) / 2)  # 默认值为范围中点

                # 使用文本输入框输入特征值
                st.markdown(f"**{feature}** (范围: {min_val:.4f} - {max_val:.4f})")
                # 使用 number_input 替代 text_input 以便更好地处理数值
                feature_values[feature] = st.number_input(
                    f"输入{feature}的值",
                    value=default_val,
                    format="%.4f", # 保持格式
                    key=f"input_{feature}"
                )

        # 预测按钮
        if st.button("获取预测结果"):
            try:
                # 检查 scaler 是否已拟合
                if not hasattr(st.session_state.data_processor, 'scaler') or not hasattr(st.session_state.data_processor.scaler, 'mean_'):
                     st.error("特征缩放器 (Scaler) 尚未拟合，请先训练模型。")
                else:
                    # 将输入的特征值转换为DataFrame，并按训练时的顺序排列
                    input_df_original = pd.DataFrame([feature_values])[st.session_state.features]

                    # 使用存储的 scaler 进行转换
                    input_df_scaled = st.session_state.data_processor.scaler.transform(input_df_original)
                    input_df_scaled = pd.DataFrame(input_df_scaled, columns=st.session_state.features) # 转换回DataFrame

                    # 使用缩放后的数据进行预测
                    prediction = st.session_state.model.predict(input_df_scaled)[0]

                    # 显示预测结果
                    st.success(f"预测结果: {prediction:.4f}")

                    # 创建包含预测结果的DataFrame用于展示
                    result_df = pd.DataFrame([feature_values])
                    result_df[st.session_state.target] = prediction # 假设目标列名已存储
                    st.dataframe(result_df)

            except Exception as e:
                error_msg = f"预测时出错: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                st.info("提示：请确保所有特征值都在有效范围内，并且格式正确。")

# 搜索推荐页面 (原特征空间定义与优化页面，已重命名)
elif page == "搜索推荐":
    st.header("搜索推荐") # 更新标题

    if st.session_state.model is None:
        st.warning("请先训练模型")
    elif st.session_state.features is None:
        st.warning("请先在'数据上传与处理'页面选择特征")
    else:
        st.info(f"当前最佳模型: {st.session_state.best_model_name}")

        # 特征空间定义
        st.subheader("特征空间定义")

        # 创建多列布局
        feature_ranges_input = {} # 使用新变量名避免覆盖会话状态
        cols = st.columns(3)

        for i, feature in enumerate(st.session_state.features):
            col_idx = i % 3
            with cols[col_idx]:
                st.markdown(f"**{feature}**")

                # 获取当前范围
                current_range = st.session_state.feature_ranges.get(feature, {})
                min_val = current_range.get("min", 0.0)
                max_val = current_range.get("max", 1.0)
                step = current_range.get("step", 0.1)

                # 设置范围和步长
                new_min = st.number_input(f"{feature} 最小值", value=float(min_val), format="%.4f", key=f"min_{feature}")
                new_max = st.number_input(f"{feature} 最大值", value=float(max_val), format="%.4f", key=f"max_{feature}")
                new_step = st.number_input(f"{feature} 步长", value=float(step), format="%.4f", min_value=0.0, key=f"step_{feature}") # 步长不能为负

                feature_ranges_input[feature] = {
                    "min": new_min,
                    "max": new_max,
                    "step": new_step
                }

        if st.button("更新特征空间并计算信息"):
            # 验证范围和步长
            valid_ranges = True
            for feature, ranges in feature_ranges_input.items():
                if ranges["min"] >= ranges["max"]:
                    st.error(f"{feature} 的最小值必须小于最大值。")
                    valid_ranges = False
                if ranges["step"] <= 0:
                    st.error(f"{feature} 的步长必须大于 0。")
                    valid_ranges = False
            
            if valid_ranges:
                st.session_state.feature_ranges = feature_ranges_input
                st.success("特征空间已更新")

                # 计算特征空间大小和预估内存消耗
                try:
                    # 获取Bootstrap模型数量（如果在会话状态中存在）
                    n_bootstraps = st.session_state.get('n_bootstraps', 50)

                    # 初始化贝叶斯优化器，不再手动设置不确定度估计方法
                    optimizer = BayesianOptimizer(
                        model=st.session_state.model,
                        feature_ranges=st.session_state.feature_ranges,
                        method=None,  # 设为None，让优化器自动选择合适的方法
                        n_bootstraps=n_bootstraps
                    )

                    # 计算特征空间大小和预估内存消耗
                    total_points, memory_mb, warning = optimizer.calculate_grid_size()

                    # 显示特征空间信息
                    st.info(f"特征空间总点数: {total_points:,}")
                    st.info(f"预估内存消耗: {memory_mb:.2f} MB")

                    # 显示内存警告信息
                    if warning:
                        st.warning(warning)
                    # 保留原有的警告逻辑作为备用
                    elif memory_mb > 1000:  # 超过1GB
                        warning_msg = "特征空间较大，可能会导致计算速度变慢或内存不足。建议增加步长或减少特征范围。"
                        if optimizer.method == 'bootstrap':
                            warning_msg += f" 当前使用Bootstrap方法({n_bootstraps}个模型)，可以考虑减少模型数量以降低内存消耗。"
                        st.warning(warning_msg)
                except Exception as e:
                    error_msg = f"计算特征空间信息时出错: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)

        # 贝叶斯优化设置
        st.subheader("贝叶斯优化设置")

        # 搜索策略选择
        search_strategy = st.selectbox(
            "选择优化搜索策略",
            ["网格搜索", "遗传算法", "粒子群优化", "模拟退火", "随机搜索"],
            index=0,
            key="search_strategy_select" # 添加key避免冲突
        )

        # 将中文策略名称映射到英文标识符
        strategy_mapping = {
            "网格搜索": "grid",
            "遗传算法": "ga",
            "粒子群优化": "pso",
            "模拟退火": "sa",
            "随机搜索": "random"
        }
        selected_strategy_key = strategy_mapping[search_strategy] # 获取映射后的策略键

        # --- 高级搜索策略参数输入 ---
        strategy_params = {}
        if selected_strategy_key == 'ga':
            st.subheader("遗传算法参数")
            cols_ga = st.columns(2)
            with cols_ga[0]:
                strategy_params['population_size'] = st.number_input("种群大小 (population_size)", min_value=10, value=50, step=10, key="ga_pop_size")
                strategy_params['crossover_prob'] = st.slider("交叉概率 (crossover_prob)", 0.0, 1.0, 0.8, 0.05, key="ga_cross_prob")
            with cols_ga[1]:
                strategy_params['n_generations'] = st.number_input("迭代代数 (n_generations)", min_value=1, value=10, step=1, key="ga_gens")
                strategy_params['mutation_prob'] = st.slider("变异概率 (mutation_prob)", 0.0, 1.0, 0.2, 0.05, key="ga_mut_prob")
        elif selected_strategy_key == 'pso':
            st.subheader("粒子群优化参数")
            cols_pso1 = st.columns(3)
            with cols_pso1[0]:
                strategy_params['n_particles'] = st.number_input("粒子数量 (n_particles)", min_value=5, value=30, step=5, key="pso_particles")
            with cols_pso1[1]:
                strategy_params['n_iterations'] = st.number_input("迭代次数 (n_iterations)", min_value=5, value=20, step=5, key="pso_iters")
            with cols_pso1[2]:
                 strategy_params['inertia_weight'] = st.number_input("惯性权重 (inertia_weight)", min_value=0.0, value=0.5, step=0.1, format="%.2f", key="pso_inertia")
            cols_pso2 = st.columns(2)
            with cols_pso2[0]:
                strategy_params['cognitive_weight'] = st.number_input("认知权重 (cognitive_weight)", min_value=0.0, value=1.5, step=0.1, format="%.2f", key="pso_cog")
            with cols_pso2[1]:
                strategy_params['social_weight'] = st.number_input("社会权重 (social_weight)", min_value=0.0, value=1.5, step=0.1, format="%.2f", key="pso_soc")
        elif selected_strategy_key == 'sa':
            st.subheader("模拟退火参数")
            cols_sa = st.columns(2)
            with cols_sa[0]:
                strategy_params['n_iterations'] = st.number_input("迭代次数 (n_iterations)", min_value=10, value=100, step=10, key="sa_iters")
                strategy_params['initial_temp'] = st.number_input("初始温度 (initial_temp)", min_value=1.0, value=100.0, step=10.0, format="%.1f", key="sa_temp")
            with cols_sa[1]:
                strategy_params['cooling_rate'] = st.slider("冷却率 (cooling_rate)", 0.8, 0.99, 0.95, 0.01, format="%.2f", key="sa_cool")
                strategy_params['n_neighbors'] = st.number_input("邻居数量 (n_neighbors)", min_value=1, value=5, step=1, key="sa_neighbors")
        # --- END ---

        # 采样函数选择
        acquisition_function = st.selectbox(
            "选择采样函数",
            ["EI (期望改进)", "UCB (置信上界)", "PI (改进概率)"],
            key="acq_func_select" # 添加key
        )

        # 显示不确定度估计方法说明
        st.info("系统将根据最优模型类型自动选择合适的不确定度估计方法：\n" +
                "- 高斯过程模型：使用模型内置的不确定度估计\n" +
                "- 随机森林模型：使用不同决策树的预测标准差\n" +
                "- 其他模型：使用Bootstrap方法估计不确定度")

        # Bootstrap模型数量设置（用于非GP和非随机森林模型）
        n_bootstraps = st.slider(
            "Bootstrap模型数量",
            10, 100, st.session_state.get('n_bootstraps', 50), 5, # 使用会话状态或默认值
            help="当使用Bootstrap方法时，更多的模型可以提供更准确的不确定度估计，但会增加计算负担和内存消耗。",
            key="n_bootstraps_slider" # 添加key
        )
        # 更新会话状态
        st.session_state.n_bootstraps = n_bootstraps

        # 推荐实验数量
        n_recommendations = st.slider("推荐实验数量", 1, 10, 3, 1, key="n_recs_slider") # 添加key

        # 优化方向
        optimization_direction = st.radio(
            "优化方向",
            ["最大化", "最小化"],
            index=0,
            key="opt_direction_radio" # 添加key
        )

        if st.button("生成实验推荐"):
            with st.spinner("正在生成实验推荐，请稍候..."):
                try:
                    # 准备原始数据 (优化器内部会处理缩放)
                    if st.session_state.data is None:
                         st.error("请先上传数据。")
                    elif not hasattr(st.session_state.data_processor, 'scaler') or not hasattr(st.session_state.data_processor.scaler, 'mean_'):
                         st.error("特征缩放器 (Scaler) 尚未拟合，请先训练模型。")
                    else:
                        X_original = st.session_state.data[st.session_state.features]
                        y = st.session_state.data[st.session_state.target]

                        # 初始化贝叶斯优化器，并传入拟合好的 scaler
                        optimizer = BayesianOptimizer(
                            model=st.session_state.model,
                            feature_ranges=st.session_state.feature_ranges,
                            acquisition_function=acquisition_function.split(" ")[0].lower(),
                            maximize=(optimization_direction == "最大化"),
                            method=None,  # 设为None，让优化器自动选择合适的方法
                            n_bootstraps=st.session_state.n_bootstraps, # 从会话状态读取
                            search_strategy=selected_strategy_key, # 使用映射后的值
                            search_strategy_params=strategy_params, # 传递策略参数
                            scaler=st.session_state.data_processor.scaler # 传入 scaler
                        )

                        # 生成推荐 (优化器内部会使用scaler)
                        recommendations = optimizer.recommend_experiments(X_original, y, n_recommendations)

                        # 保存到会话状态
                        st.session_state.recommendations = recommendations

                        # 显示推荐
                        st.subheader("实验推荐")
                        st.dataframe(recommendations)

                        # 提供下载选项
                        csv = recommendations.to_csv(index=False)
                        st.download_button(
                            label="下载推荐实验条件",
                            data=csv,
                            file_name="experiment_recommendations.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    error_msg = f"生成实验推荐时出错: {str(e)}"
                    logger.error(error_msg, exc_info=True)  # 记录完整错误信息到日志
                    st.error(error_msg)
                    st.error("详细错误信息已记录到日志文件")

# 迭代优化页面
elif page == "迭代优化":
    st.header("迭代优化")

    if st.session_state.model is None:
        st.warning("请先训练模型")
    elif st.session_state.features is None or st.session_state.target is None:
         st.warning("请先在'数据上传与处理'页面选择特征和目标")
    else:
        st.info(f"当前最佳模型: {st.session_state.best_model_name}")

        # 显示优化历史
        if len(st.session_state.optimization_history) > 0:
            st.subheader("优化历史")

            # 创建历史数据表
            history_data = []
            for i, history in enumerate(st.session_state.optimization_history):
                history_data.append({
                    "迭代": i + 1,
                    "数据量": history["data_size"],
                    "最佳模型": history["best_model"],
                    "最佳性能": f"{history['best_score']:.4f}",
                    "目标最优值": f"{history['best_target']:.4f}"
                })

            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df)

            # 绘制优化进程图
            st.subheader("优化进程")

            # 提取数据用于绘图
            iterations = [i+1 for i in range(len(st.session_state.optimization_history))]
            best_targets = [h["best_target"] for h in st.session_state.optimization_history]

            # 创建图表数据
            chart_data = pd.DataFrame({
                "迭代": iterations,
                "目标最优值": best_targets
            })

            # 绘制折线图
            st.line_chart(chart_data.set_index("迭代"))

        # 上传新实验数据
        st.subheader("上传新实验数据")

        uploaded_file = st.file_uploader("上传新实验数据CSV文件", type=["csv"], key="iter_upload") # 添加key

        if uploaded_file is not None:
            try:
                # 读取新数据
                new_data = pd.read_csv(uploaded_file)

                # 检查数据格式
                required_columns = st.session_state.features + [st.session_state.target]
                missing_columns = [col for col in required_columns if col not in new_data.columns]

                if missing_columns:
                    st.error(f"新数据缺少以下列: {', '.join(missing_columns)}")
                else:
                    # 显示新数据预览
                    st.subheader("新数据预览")
                    st.dataframe(new_data[required_columns])

                    if st.button("合并数据并更新模型"):
                        with st.spinner("正在更新模型，请稍候..."):
                            # 合并数据
                            combined_data = pd.concat([st.session_state.data, new_data[required_columns]], ignore_index=True)

                            # 更新会话状态中的数据
                            st.session_state.data = combined_data

                            # 准备合并后的数据
                            X_combined_original = combined_data[st.session_state.features]
                            y_combined = combined_data[st.session_state.target]

                            # 在合并后的数据上重新拟合 Scaler
                            logger.info("在合并后的数据上重新拟合 Scaler...")
                            X_combined_scaled = st.session_state.data_processor.scale_features(X_combined_original.copy(), st.session_state.features)
                            logger.info(f"Scaler 已在合并数据上重新拟合: {st.session_state.data_processor.scaler}")

                            # 重新训练模型 (使用缩放后的数据)
                            # 需要重新初始化 ModelTrainer 以便使用新的模型列表和参数（如果界面允许修改的话）
                            # 假设模型选择和参数与上次训练相同
                            models_to_train_iter = st.session_state.get('models_to_train', ["Lasso", "随机森林", "XGBoost", "SVR", "高斯过程"]) # 获取上次选择或默认
                            metric_iter = st.session_state.get('metric', 'r2')
                            cv_folds_iter = st.session_state.get('cv_folds', 5)
                            test_size_iter = st.session_state.get('test_size', 0.2)

                            trainer = ModelTrainer(
                                models=models_to_train_iter,
                                metric=metric_iter.lower(),
                                cv_folds=cv_folds_iter,
                                test_size=test_size_iter
                            )
                            logger.info("开始在合并数据上重新训练模型...")
                            results = trainer.train_and_evaluate(X_combined_scaled, y_combined)
                            logger.info("模型重新训练完成。")

                            # 获取最佳模型
                            results_df = pd.DataFrame(results).sort_values(
                                by="cv_score",
                                ascending=True if metric_iter.lower() == "rmse" else False
                            )
                            best_model_name = results_df.iloc[0]["model_name"]
                            best_model = trainer.get_best_model()

                            # 更新会话状态
                            st.session_state.model = best_model
                            st.session_state.best_model_name = best_model_name

                            # 获取当前最佳目标值 (在原始尺度上)
                            optimization_direction_iter = st.session_state.get('optimization_direction', "最大化") # 获取上次选择或默认
                            best_target = y_combined.max() if optimization_direction_iter == "最大化" else y_combined.min()

                            # 更新优化历史
                            st.session_state.optimization_history.append({
                                "data_size": len(combined_data),
                                "best_model": best_model_name,
                                "best_score": float(results_df.iloc[0]["cv_score"]), # 使用 CV 分数记录性能
                                "best_target": float(best_target)
                            })

                            st.success(f"模型已更新！当前数据集大小: {len(combined_data)}")
                            st.info(f"新的最佳模型: {best_model_name}")
                            st.info(f"特征缩放器 (Scaler) 已在合并数据上重新拟合。")

                            # 提示用户下一步操作
                            st.info("请前往'搜索推荐'页面生成新的实验推荐")

            except Exception as e:
                error_msg = f"处理新数据时出错: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)

# 全局异常处理，确保所有未捕获的异常都被记录
try:
    # 主程序已经在上面运行完毕，这里只是为了捕获全局异常
    pass
except Exception as e:
    error_msg = f"应用运行时发生未捕获的异常: {str(e)}"
    logger.error(error_msg, exc_info=True)
    st.error(error_msg)


# 页面底部信息
st.markdown("---")
st.markdown("© 2025 贝叶斯优化材料组分系统 | 基于BO的材料优化平台 | develop by TTRS-SH ATRD")
