# 导入必要的库
import streamlit as st  # Streamlit用于构建Web应用界面
import pandas as pd     # Pandas用于数据处理和分析
import numpy as np      # NumPy用于数值计算
import os               # 操作系统接口
import sys              # 系统相关功能
import logging          # 日志记录
import traceback        # 异常跟踪
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
logger.info("贝叶斯优化材料组分系统启动中...")

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

# 主页面标题
st.title("贝叶斯优化材料组分系统")

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择功能页面",
    ["数据上传与处理", "模型训练与评估", "特征空间定义与优化", "迭代优化"]
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
        
        # 搜索策略选择
        st.subheader("搜索策略选择")
        search_strategy = st.selectbox(
            "选择优化搜索策略",
            ["网格搜索", "遗传算法", "粒子群优化", "模拟退火", "随机搜索"],
            index=0
        )
        
        # 将中文策略名称映射到英文标识符
        strategy_mapping = {
            "网格搜索": "grid",
            "遗传算法": "ga",
            "粒子群优化": "pso",
            "模拟退火": "sa",
            "随机搜索": "random"
        }
        st.session_state.search_strategy = strategy_mapping[search_strategy]
        
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
                    X = st.session_state.data[st.session_state.features]
                    y = st.session_state.data[st.session_state.target]
                    
                    # 初始化模型训练器
                    trainer = ModelTrainer(
                        models=models_to_train,
                        metric=metric.lower(),
                        cv_folds=cv_folds,
                        test_size=test_size
                    )
                    
                    # 训练模型
                    results = trainer.train_and_evaluate(X, y)
                    
                    # 显示结果
                    st.subheader("模型评估结果")
                    
                    # 创建结果数据框
                    # 注意：这里应该使用cv_score而不是test_score来保持与ModelTrainer中选择最佳模型的逻辑一致
                    results_df = pd.DataFrame(results).sort_values(
                        by="cv_score", 
                        ascending=False if metric == "RMSE" else True
                    )
                    
                    # 格式化结果
                    results_df["test_score"] = results_df["test_score"].apply(lambda x: f"{x:.4f}")
                    results_df["cv_score"] = results_df["cv_score"].apply(lambda x: f"{x:.4f}")
                    results_df["train_score"] = results_df["train_score"].apply(lambda x: f"{x:.4f}")
                    
                    # 显示结果表格
                    # 创建格式化后的显示数据框
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
                    
                except Exception as e:
                    error_msg = f"模型训练时出错: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)

# 特征空间定义与优化页面
elif page == "特征空间定义与优化":
    st.header("特征空间定义与优化")
    
    if st.session_state.model is None:
        st.warning("请先训练模型")
    else:
        st.info(f"当前最佳模型: {st.session_state.best_model_name}")
        
        # 特征空间定义
        st.subheader("特征空间定义")
        
        # 创建多列布局
        feature_ranges = {}
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
                new_min = st.number_input(f"{feature} 最小值", value=float(min_val), format="%.4f")
                new_max = st.number_input(f"{feature} 最大值", value=float(max_val), format="%.4f")
                new_step = st.number_input(f"{feature} 步长", value=float(step), format="%.4f")
                
                feature_ranges[feature] = {
                    "min": new_min,
                    "max": new_max,
                    "step": new_step
                }
        
        if st.button("更新特征空间"):
            st.session_state.feature_ranges = feature_ranges
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
        
        # 自主填写特征值并获取预测结果
        st.subheader("特征值预测")
        st.markdown("在下方输入特征值，获取预测结果")
        
        # 创建多列布局用于输入特征值
        if st.session_state.features:
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
                    feature_values[feature] = float(st.text_input(
                        f"输入{feature}的值",
                        value=f"{default_val:.4f}",
                        key=f"input_{feature}"
                    ))
            
            # 预测按钮
            if st.button("获取预测结果"):
                if st.session_state.model is None:
                    st.error("请先训练模型")
                else:
                    try:
                        # 验证输入值是否为有效数字
                        valid_input = True
                        for feature, value in feature_values.items():
                            try:
                                feature_values[feature] = float(value)
                            except (ValueError, TypeError):
                                st.error(f"{feature}的输入值'{value}'不是有效的数字")
                                valid_input = False
                                break
                        
                        if valid_input:
                            # 获取Bootstrap模型数量（如果在会话状态中存在）
                            n_bootstraps = st.session_state.get('n_bootstraps', 50)
                            
                            # 初始化贝叶斯优化器，不再手动设置不确定度估计方法
                            optimizer = BayesianOptimizer(
                                model=st.session_state.model,
                                feature_ranges=st.session_state.feature_ranges,
                                method=None,  # 设为None，让优化器自动选择合适的方法
                                n_bootstraps=n_bootstraps,
                                search_strategy=st.session_state.search_strategy
                            )
                            
                            # 获取预测结果
                            prediction = optimizer.predict_for_features(feature_values)
                            
                            # 显示预测结果
                            st.success(f"预测结果: {prediction:.4f}")
                            
                            # 创建包含预测结果的DataFrame用于展示
                            result_df = pd.DataFrame([feature_values])
                            result_df[st.session_state.target] = prediction
                            st.dataframe(result_df)
                    except Exception as e:
                        st.error(f"预测时出错: {str(e)}")
                        st.info("提示：请确保所有特征值都在有效范围内，并且格式正确。")
        
        # 贝叶斯优化设置
        st.subheader("贝叶斯优化设置")
        
        # 搜索策略选择
        search_strategy = st.selectbox(
            "选择搜索策略",
            ["网格搜索", "随机搜索", "贝叶斯优化"],
            index=0
        )
        
        # 采样函数选择
        acquisition_function = st.selectbox(
            "选择采样函数",
            ["EI (期望改进)", "UCB (置信上界)", "PI (改进概率)"]
        )
        
        # 显示不确定度估计方法说明
        st.info("系统将根据最优模型类型自动选择合适的不确定度估计方法：\n" +
                "- 高斯过程模型：使用模型内置的不确定度估计\n" +
                "- 随机森林模型：使用不同决策树的预测标准差\n" +
                "- 其他模型：使用Bootstrap方法估计不确定度")
        
        # Bootstrap模型数量设置（用于非GP和非随机森林模型）
        n_bootstraps = st.slider(
            "Bootstrap模型数量", 
            10, 100, 50, 5,
            help="当使用Bootstrap方法时，更多的模型可以提供更准确的不确定度估计，但会增加计算负担和内存消耗。"
        )
        
        # 推荐实验数量
        n_recommendations = st.slider("推荐实验数量", 1, 10, 3, 1)
        
        # 优化方向
        optimization_direction = st.radio(
            "优化方向",
            ["最大化", "最小化"],
            index=0
        )
        
        # 保存Bootstrap模型数量到会话状态
        st.session_state.n_bootstraps = n_bootstraps
        
        if st.button("生成实验推荐"):
            with st.spinner("正在生成实验推荐，请稍候..."):
                try:
                    # 准备数据
                    X = st.session_state.data[st.session_state.features]
                    y = st.session_state.data[st.session_state.target]
                    
                    # 初始化贝叶斯优化器
                    # 不再手动设置不确定度估计方法，而是让优化器自动选择
                    optimizer = BayesianOptimizer(
                        model=st.session_state.model,
                        feature_ranges=st.session_state.feature_ranges,
                        acquisition_function=acquisition_function.split(" ")[0].lower(),
                        maximize=(optimization_direction == "最大化"),
                        method=None,  # 设为None，让优化器自动选择合适的方法
                        n_bootstraps=n_bootstraps,
                        search_strategy=st.session_state.search_strategy
                    )
                    
                    # 生成推荐
                    recommendations = optimizer.recommend_experiments(X, y, n_recommendations)
                    
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
                    st.error(f"生成实验推荐时出错: {str(e)}")

# 迭代优化页面
elif page == "迭代优化":
    st.header("迭代优化")
    
    if st.session_state.model is None:
        st.warning("请先训练模型")
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
        
        uploaded_file = st.file_uploader("上传新实验数据CSV文件", type=["csv"])
        
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
                            
                            # 准备数据
                            X = combined_data[st.session_state.features]
                            y = combined_data[st.session_state.target]
                            
                            # 重新训练模型
                            trainer = ModelTrainer()
                            results = trainer.train_and_evaluate(X, y)
                            
                            # 获取最佳模型
                            # 注意：这里应该使用cv_score而不是test_score来保持与ModelTrainer中选择最佳模型的逻辑一致
                            results_df = pd.DataFrame(results).sort_values(by="cv_score", ascending=False if st.session_state.metric == "RMSE" else True)
                            best_model_name = results_df.iloc[0]["model_name"]
                            best_model = trainer.get_best_model()
                            
                            # 更新会话状态
                            st.session_state.model = best_model
                            st.session_state.best_model_name = best_model_name
                            
                            # 获取当前最佳目标值
                            best_target = y.max() if st.session_state.optimization_direction == "最大化" else y.min()
                            
                            # 保存优化方向
                            if 'optimization_direction' not in st.session_state:
                                st.session_state.optimization_direction = "最大化"  # 默认为最大化
                                
                            # 更新优化历史
                            st.session_state.optimization_history.append({
                                "data_size": len(combined_data),
                                "best_model": best_model_name,
                                "best_score": float(results_df.iloc[0]["test_score"]),
                                "best_target": float(best_target)
                            })
                            
                            st.success(f"模型已更新！当前数据集大小: {len(combined_data)}")
                            st.info(f"新的最佳模型: {best_model_name}")
                            
                            # 重定向到特征空间定义与优化页面
                            st.info("请前往'特征空间定义与优化'页面生成新的实验推荐")
            
            except Exception as e:
                st.error(f"处理新数据时出错: {str(e)}")    

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
st.markdown("© 2025 贝叶斯优化材料组分系统 | 基于BO的材料优化平台")