import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入自定义模块
from app.utils.data_processor import DataProcessor
from app.models.model_trainer import ModelTrainer
from app.optimization.bayesian_optimizer import BayesianOptimizer

# 设置页面配置
st.set_page_config(
    page_title="贝叶斯优化材料组分系统",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'feature_ranges' not in st.session_state:
    st.session_state.feature_ranges = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []

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
    st.header("数据上传与处理")
    
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # 读取数据
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(data.head())
            
            # 选择特征和目标列
            st.subheader("选择特征和目标列")
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
            st.error(f"处理数据时出错: {str(e)}")

# 模型训练与评估页面
elif page == "模型训练与评估":
    st.header("模型训练与评估")
    
    if st.session_state.data is None or st.session_state.features is None or st.session_state.target is None:
        st.warning("请先上传数据并选择特征和目标列")
    else:
        st.info(f"当前数据集: {st.session_state.data.shape[0]} 行, {st.session_state.data.shape[1]} 列")
        st.info(f"选择的特征: {', '.join(st.session_state.features)}")
        st.info(f"选择的目标: {st.session_state.target}")
        
        # 模型选择
        st.subheader("模型选择")
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
                    results_df = pd.DataFrame(results).sort_values(
                        by="test_score", 
                        ascending=False if metric == "RMSE" else True
                    )
                    
                    # 格式化结果
                    results_df["test_score"] = results_df["test_score"].apply(lambda x: f"{x:.4f}")
                    results_df["cv_score"] = results_df["cv_score"].apply(lambda x: f"{x:.4f}")
                    results_df["train_score"] = results_df["train_score"].apply(lambda x: f"{x:.4f}")
                    
                    # 显示结果表格
                    st.dataframe(results_df)
                    
                    # 获取最佳模型
                    best_model_name = results_df.iloc[0]["model_name"]
                    best_model = trainer.get_best_model()
                    
                    # 保存到会话状态
                    st.session_state.model = best_model
                    st.session_state.best_model_name = best_model_name
                    
                    st.success(f"模型训练完成！最佳模型: {best_model_name}")
                    
                except Exception as e:
                    st.error(f"模型训练时出错: {str(e)}")

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
        
        # 贝叶斯优化设置
        st.subheader("贝叶斯优化设置")
        
        # 采样函数选择
        acquisition_function = st.selectbox(
            "选择采样函数",
            ["EI (期望改进)", "UCB (置信上界)", "PI (改进概率)"]
        )
        
        # 推荐实验数量
        n_recommendations = st.slider("推荐实验数量", 1, 10, 3, 1)
        
        # 优化方向
        optimization_direction = st.radio(
            "优化方向",
            ["最大化", "最小化"],
            index=0
        )
        
        if st.button("生成实验推荐"):
            with st.spinner("正在生成实验推荐，请稍候..."):
                try:
                    # 准备数据
                    X = st.session_state.data[st.session_state.features]
                    y = st.session_state.data[st.session_state.target]
                    
                    # 初始化贝叶斯优化器
                    optimizer = BayesianOptimizer(
                        model=st.session_state.model,
                        feature_ranges=st.session_state.feature_ranges,
                        acquisition_function=acquisition_function.split(" ")[0].lower(),
                        maximize=(optimization_direction == "最大化")
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
                            results_df = pd.DataFrame(results).sort_values(by="test_score", ascending=False)
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

# 页面底部信息
st.markdown("---")
st.markdown("© 2023 贝叶斯优化材料组分系统 | 基于Docker的材料优化平台")