import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.utils import resample
from app.optimization.search_strategies import (
    SearchStrategy, GridSearch, GeneticAlgorithm,
    ParticleSwarmOptimization, SimulatedAnnealing, RandomSearch
)
# Note: Duplicate import removed below
# from app.optimization.search_strategies import (
#     SearchStrategy, GridSearch, GeneticAlgorithm,
#     ParticleSwarmOptimization, SimulatedAnnealing, RandomSearch
# )

from sklearn.preprocessing import StandardScaler # 导入 StandardScaler 用于类型检查

class BayesianOptimizer:
    """
     贝叶斯优化器类，用于特征空间搜索和实验推荐
     """

    def __init__(self, model=None, feature_ranges=None, acquisition_function='ei', maximize=True, random_state=42, method=None, n_bootstraps=50, search_strategy='grid', search_strategy_params=None, scaler=None, max_sum=None): # 添加 max_sum 参数
        """
        初始化贝叶斯优化器

        参数:
            model: 训练好的模型，用于预测
            feature_ranges: 特征范围字典，格式为 {feature_name: {'min': min_val, 'max': max_val, 'step': step_val}}
            acquisition_function: 采样函数，可选 'ei'(期望改进), 'ucb'(置信上界), 'pi'(改进概率)
            maximize: 是否最大化目标，True为最大化，False为最小化
            random_state: 随机种子
            method: 不确定度估计方法，可选 'gp'(高斯过程), 'rf'(随机森林), 或 'bootstrap'(Bootstrap方法)
                   如果为None，则根据模型类型自动选择合适的方法
            n_bootstraps: 使用Bootstrap方法或随机森林方法时的模型/树数量
            search_strategy: 特征空间搜索策略，可选 'grid'(网格搜索), 'ga'(遗传算法), 'pso'(粒子群优化),
                             'sa'(模拟退火), 'random'(随机搜索)，或者直接传入SearchStrategy的实例
             search_strategy_params: 包含高级搜索策略参数的字典 (可选)
             scaler: 用于特征缩放的 scikit-learn scaler 对象 (例如 StandardScaler)
             max_sum: 特征和的最大值约束，如果为None则不应用约束
         """
        self.model = model
        self.feature_ranges = feature_ranges
        self.acquisition_function = acquisition_function.lower()
        self.maximize = maximize # Fix indentation
        self.random_state = random_state # Fix indentation
        self.gp_model = None
        self.n_bootstraps = n_bootstraps
        self.bootstrap_models = [] # Fix indentation
        self.search_strategy_params = search_strategy_params if isinstance(search_strategy_params, dict) else {} # Fix indentation
        self.scaler = scaler # Fix indentation
        self.max_sum = max_sum # 保存约束参数

        # --- START FEATURE COUNT VALIDATION ---
        if not isinstance(feature_ranges, dict):
            raise TypeError("feature_ranges 必须是字典类型")
        num_features_in_ranges = len(feature_ranges)
        if num_features_in_ranges == 0:
            raise ValueError("feature_ranges 不能为空字典")

        # Check consistency with the provided model, if applicable
        if model is not None and hasattr(model, 'n_features_in_'):
            if model.n_features_in_ != num_features_in_ranges:
                raise ValueError(f"模型期望的特征数量 ({model.n_features_in_}) 与 feature_ranges 中定义的数量 ({num_features_in_ranges}) 不匹配。")
        # --- END FEATURE COUNT VALIDATION ---

        # 自动选择不确定度估计方法
        if method is None:
            self.method = self._auto_select_uncertainty_method()
        else:
            self.method = method.lower()

        # 验证采样函数 (Moved validation here)
        valid_acq_funcs = ['ei', 'ucb', 'pi']
        if self.acquisition_function not in valid_acq_funcs:
            raise ValueError(f"不支持的采样函数: {self.acquisition_function}，可选: {', '.join(valid_acq_funcs)}")

        # 验证不确定度估计方法 (Moved validation here)
        valid_methods = ['gp', 'rf', 'bootstrap']
        if self.method not in valid_methods:
            raise ValueError(f"不支持的不确定度估计方法: {self.method}，可选: {', '.join(valid_methods)}")

        # 设置搜索策略
        self.search_strategy = self._create_search_strategy(search_strategy)

    def _auto_select_uncertainty_method(self):
        """
        根据模型类型自动选择合适的不确定度估计方法

        返回:
            str: 选择的不确定度估计方法 ('gp', 'rf', 或 'bootstrap')
        """
        if self.model is None:
            # 如果没有模型，默认使用高斯过程
            return 'gp'

        # 获取模型类型的字符串表示
        model_type = str(type(self.model))

        # 判断模型类型
        if 'GaussianProcessRegressor' in model_type:
            # 高斯过程模型使用其内置的不确定度估计
            return 'gp'
        elif 'RandomForestRegressor' in model_type:
            # 随机森林使用不同estimator的标准差
            return 'rf'
        else:
            # 其他模型使用bootstrap方法
            return 'bootstrap'

    def _create_search_strategy(self, search_strategy):
        """
        创建搜索策略实例

        参数:
            search_strategy: 搜索策略名称或实例

        返回:
            SearchStrategy实例
        """
        # 如果已经是SearchStrategy实例，直接返回
        if isinstance(search_strategy, SearchStrategy):
            return search_strategy

        # 根据名称创建相应的搜索策略实例
        if isinstance(search_strategy, str):
            strategy_name = search_strategy.lower()

            if strategy_name == 'grid':
                return GridSearch(self.feature_ranges, self.random_state, self.max_sum)
            elif strategy_name == 'ga':
                # 使用 ** 解包传递参数，如果参数字典中没有对应键，则使用类定义的默认值
                return GeneticAlgorithm(self.feature_ranges, self.random_state, max_sum=self.max_sum, **self.search_strategy_params)
            elif strategy_name == 'pso':
                return ParticleSwarmOptimization(self.feature_ranges, self.random_state, max_sum=self.max_sum, **self.search_strategy_params)
            elif strategy_name == 'sa':
                return SimulatedAnnealing(self.feature_ranges, self.random_state, max_sum=self.max_sum, **self.search_strategy_params)
            elif strategy_name == 'random':
                return RandomSearch(self.feature_ranges, self.random_state, self.max_sum)
            else:
                raise ValueError(f"不支持的搜索策略: {search_strategy}，可选: 'grid', 'ga', 'pso', 'sa', 'random'")

        # 如果不是字符串也不是SearchStrategy实例，抛出异常
        raise ValueError("search_strategy必须是字符串或SearchStrategy实例")

    def calculate_grid_size(self):
        """
        计算特征空间网格的大小，并估算内存消耗

        返回:
            tuple: (总点数, 预估内存消耗(MB), 警告信息)
        """
        if not self.feature_ranges:
            raise ValueError("未设置特征范围")

        # 根据搜索策略估算内存消耗
        if hasattr(self, 'search_strategy') and self.search_strategy is not None:
            # 使用搜索策略的内存估算方法
            # 对于网格搜索，计算总点数
            if isinstance(self.search_strategy, GridSearch):
                # 计算每个特征的点数
                points_per_feature = []
                for feature, range_info in self.feature_ranges.items():
                    min_val = range_info['min']
                    max_val = range_info['max']

                    if 'step' in range_info and range_info['step'] > 0:
                        # 使用指定的步长
                        step = range_info['step']
                        num_points = int(np.ceil((max_val - min_val) / step)) + 1
                    else:
                        # 默认使用10个点
                        num_points = 10

                    points_per_feature.append(num_points)

                # 计算总点数
                total_points = np.prod(points_per_feature)
            else:
                # 对于其他搜索策略，使用估计值
                # 遗传算法: 种群大小 * 迭代次数
                if isinstance(self.search_strategy, GeneticAlgorithm):
                    total_points = self.search_strategy.population_size * self.search_strategy.n_generations
                # 粒子群优化: 粒子数量 * 迭代次数
                elif isinstance(self.search_strategy, ParticleSwarmOptimization):
                    total_points = self.search_strategy.n_particles * self.search_strategy.n_iterations
                # 模拟退火: 迭代次数 * 邻居数量
                elif isinstance(self.search_strategy, SimulatedAnnealing):
                    total_points = self.search_strategy.n_iterations * self.search_strategy.n_neighbors
                # 随机搜索: 采样点数量
                elif isinstance(self.search_strategy, RandomSearch):
                    # Check if n_samples attribute exists, otherwise use a default
                    total_points = getattr(self.search_strategy, 'n_samples', 1000) # Use getattr for safety
                else:
                    # 默认估计值
                    total_points = 1000

            # 使用搜索策略的内存估算方法
            memory_mb, warning = self.search_strategy.calculate_memory_usage(total_points)

            # 添加不确定度估计方法的内存消耗
            if self.method == 'bootstrap':
                warning += f"\n当前使用Bootstrap方法({self.n_bootstraps}个模型实例)，可能会增加额外的内存消耗。"

            return total_points, memory_mb, warning
        else:
            # 如果没有设置搜索策略，使用传统的网格搜索内存估算
            # 计算每个特征的点数
            points_per_feature = []
            for feature, range_info in self.feature_ranges.items():
                min_val = range_info['min']
                max_val = range_info['max']

                if 'step' in range_info and range_info['step'] > 0:
                    # 使用指定的步长
                    step = range_info['step']
                    num_points = int(np.ceil((max_val - min_val) / step)) + 1
                else:
                    # 默认使用10个点
                    num_points = 10

                points_per_feature.append(num_points)

            # 计算总点数
            total_points = np.prod(points_per_feature)

            # 估算数据存储内存消耗 (每个浮点数8字节，再加上DataFrame的开销)
            # 假设每个点有len(feature_ranges)个特征，加上2个额外列(预测值和采样函数值)
            num_columns = len(self.feature_ranges) + 2
            data_memory_bytes = total_points * num_columns * 8  # 8字节/浮点数

            # 根据不确定度估计方法估算内存消耗
            if self.method == 'gp':
                # 估算高斯过程计算中的内存消耗
                # 1. 协方差矩阵: O(n²) 大小，其中n是总点数
                cov_matrix_bytes = total_points * total_points * 8

                # 2. 中间计算数组 (如预测均值、标准差等)
                intermediate_arrays_bytes = total_points * 3 * 8  # 假设有3个主要中间数组

                # 3. 高斯过程模型内部使用的其他数组
                gp_overhead_bytes = total_points * 5 * 8  # 估计值

                # 高斯过程方法的额外内存消耗
                method_memory_bytes = cov_matrix_bytes + intermediate_arrays_bytes + gp_overhead_bytes

            elif self.method == 'bootstrap' or self.method == 'rf':
                # 估算Bootstrap方法或随机森林方法的内存消耗
                # 1. n_bootstraps个模型实例的内存开销
                # 假设每个模型实例的大小与数据集大小成正比
                model_instance_bytes = len(self.feature_ranges) * total_points * 8 * 0.5  # 估计每个模型实例大小
                bootstrap_models_bytes = self.n_bootstraps * model_instance_bytes

                # 2. 预测时的额外数组开销 (n_bootstraps个预测结果数组)
                prediction_arrays_bytes = self.n_bootstraps * total_points * 8

                # 3. 计算均值和标准差的中间数组
                bootstrap_intermediate_bytes = total_points * 2 * 8  # 均值和标准差数组

                # Bootstrap方法或随机森林方法的额外内存消耗
                method_memory_bytes = bootstrap_models_bytes + prediction_arrays_bytes + bootstrap_intermediate_bytes

            else:
                # 默认使用高斯过程的内存估算
                cov_matrix_bytes = total_points * total_points * 8
                intermediate_arrays_bytes = total_points * 3 * 8
                gp_overhead_bytes = total_points * 5 * 8
                method_memory_bytes = cov_matrix_bytes + intermediate_arrays_bytes + gp_overhead_bytes

            # 总内存消耗
            total_memory_bytes = data_memory_bytes + method_memory_bytes
            total_memory_mb = total_memory_bytes / (1024 * 1024)  # 转换为MB

            # 生成警告信息
            warning = ""
            if total_memory_mb > 8000:  # 超过8GB内存
                warning = "警告: 预估内存消耗超过8GB，可能导致系统不稳定或崩溃。建议减小步长或使用批处理方式。"
                if self.method == 'bootstrap':
                    warning += f" 当前使用Bootstrap方法({self.n_bootstraps}个模型实例)，可以考虑减少n_bootstraps参数值。"
            elif total_memory_mb > 4000:  # 超过4GB内存
                warning = "警告: 预估内存消耗较大(>4GB)，可能影响系统性能。考虑减小步长以降低内存使用。"
                if self.method == 'bootstrap':
                    warning += f" 使用Bootstrap方法时，n_bootstraps参数({self.n_bootstraps})会显著影响内存使用。"
            elif total_memory_mb > 2000:  # 超过2GB内存
                warning = "提示: 预估内存消耗中等(>2GB)，在低配置机器上可能影响性能。"

            return total_points, total_memory_mb, warning

    def _generate_grid(self):
        """
        生成特征空间网格

        返回:
            特征网格点的DataFrame
        """
        if not self.feature_ranges:
            raise ValueError("未设置特征范围")

        grid_points = []
        # Use self.feature_names which is set in __init__
        feature_names = self.feature_names

        # 为每个特征生成网格点
        for feature in feature_names:
            range_info = self.feature_ranges[feature]
            min_val = range_info['min']
            max_val = range_info['max']

            if 'step' in range_info and range_info['step'] > 0:
                # 使用指定的步长
                step = range_info['step']
                points = np.arange(min_val, max_val + step/2, step)  # 加step/2确保包含max_val
            else:
                # 默认使用10个点
                points = np.linspace(min_val, max_val, 10)

            grid_points.append(points)

        # --- START PRE-MESHGRID MEMORY CHECK ---
        total_expected_points = np.prod([len(p) for p in grid_points])
        num_features = len(feature_names)
        # Estimate required memory for the grid DataFrame (float64 = 8 bytes)
        # This is a lower bound, meshgrid itself might need more temporarily
        required_mem_gb = (total_expected_points * num_features * 8) / (1024**3)
        MEMORY_THRESHOLD_GB = 8 # Define a threshold (e.g., 8 GB)

        if required_mem_gb > MEMORY_THRESHOLD_GB:
             raise MemoryError(f"预估网格内存 ({required_mem_gb:.2f} GB) 超过阈值 ({MEMORY_THRESHOLD_GB} GB)。预期点数: {total_expected_points}, 特征数: {num_features}. 请减小特征范围的 'step' 或默认分辨率，或切换搜索策略。")
        # --- END PRE-MESHGRID MEMORY CHECK ---

        # 生成网格
        try:
            mesh = np.meshgrid(*grid_points, indexing='ij')
            # 修正网格生成逻辑
            grid = np.vstack([m.ravel() for m in mesh]).T
        except MemoryError as e:
             # Catch memory error specifically during meshgrid/vstack
             # Recalculate details for the error message
             total_expected_points_err = np.prod([len(p) for p in grid_points])
             num_features_err = len(feature_names)
             required_mem_gb_err = (total_expected_points_err * num_features_err * 8) / (1024**3)
             raise MemoryError(f"无法在 np.meshgrid/vstack 中为网格分配内存。预期点数: {total_expected_points_err}, 特征数: {num_features_err}, 预估内存: {required_mem_gb_err:.2f} GB. 请减小特征范围的 'step' 或默认分辨率，或切换搜索策略。原始错误: {e}")
        except Exception as e:
             # Catch other potential errors during grid generation
             raise RuntimeError(f"生成网格时出错: {e}")

        # 转换为DataFrame
        grid_df = pd.DataFrame(grid, columns=feature_names)

        return grid_df

    def _fit_gp(self, X, y):
        """
        拟合高斯过程模型

        参数:
            X: 特征矩阵
            y: 目标向量
        """
        # 如果需要最小化，则取负值
        y_opt = y if self.maximize else -y

        # 如果有 scaler，则对 X 进行转换
        X_scaled = self.scaler.transform(X) if self.scaler else X

        # 创建高斯过程回归模型
        kernel = ConstantKernel() * Matern(nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )

        # 使用缩放后的数据拟合模型
        self.gp_model.fit(X_scaled, y_opt)

    def _fit_bootstrap(self, X, y):
        """
        使用Bootstrap方法拟合多个模型实例

        参数:
            X: 特征矩阵
            y: 目标向量
        """
        # 如果需要最小化，则取负值
        y_opt = y if self.maximize else -y

        # 如果有 scaler，则对 X 进行转换
        X_scaled = self.scaler.transform(X) if self.scaler else X

        # 清空之前的Bootstrap模型
        self.bootstrap_models = []

        # 创建随机数生成器
        rng = np.random.RandomState(self.random_state)

        # 创建n_bootstraps个模型实例并在不同的数据子集上训练
        for i in range(self.n_bootstraps):
            # 使用Bootstrap重采样生成训练数据 (在缩放后的数据上采样)
            X_boot_scaled, y_boot = resample(X_scaled, y_opt, random_state=rng.randint(0, 10000))

            # 克隆原始模型
            if self.model is None:
                raise ValueError("使用Bootstrap方法时必须提供基础模型")

            # 尝试使用sklearn的clone函数克隆模型
            try:
                from sklearn.base import clone
                boot_model = clone(self.model)
            except (ImportError, TypeError):
                # 如果不能使用clone，则尝试直接复制模型的类
                try:
                    model_class = self.model.__class__
                    boot_model = model_class(**self.model.get_params())
                except AttributeError:
                    # 如果上述方法都失败，则提示用户提供可克隆的模型
                    raise ValueError("无法克隆提供的模型，请确保模型支持克隆或get_params方法")

            # 使用缩放后的数据训练模型
            boot_model.fit(X_boot_scaled, y_boot)

            # 添加到Bootstrap模型列表
            self.bootstrap_models.append(boot_model)

    def _predict_bootstrap(self, X):
        """
        使用Bootstrap模型进行预测，并计算预测的均值和标准差
        """
        if not self.bootstrap_models:
            raise ValueError("Bootstrap模型未拟合，请先调用_fit_bootstrap方法")

        # 如果有 scaler，则对 X 进行转换
        X_scaled = self.scaler.transform(X) if self.scaler else X

        # 收集所有模型的预测结果 (在缩放后的数据上预测)
        predictions = np.array([model.predict(X_scaled).flatten() for model in self.bootstrap_models]) # 添加flatten()展平结果

        # 确保predictions是正确的形状
        if len(predictions.shape) > 2:
            predictions = predictions.reshape(self.n_bootstraps, -1)

        # 计算预测均值和标准差
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)

        # 确保返回的sigma是一维数组
        sigma = sigma.flatten()  # 添加额外的展平操作

        return mu, sigma

    def _predict_random_forest(self, X):
        """
        使用随机森林模型进行预测，并使用不同estimator的预测值计算标准差

        参数:
            X: 待预测的特征矩阵

        返回:
            tuple: (预测均值, 预测标准差)
        """
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("模型不是随机森林或没有estimators_属性")

        # 如果有 scaler，则对 X 进行转换
        X_scaled = self.scaler.transform(X) if self.scaler else X

        # 获取随机森林的所有决策树
        estimators = self.model.estimators_

        # 收集所有决策树的预测结果 (在缩放后的数据上预测)
        predictions = np.array([estimator.predict(X_scaled).flatten() for estimator in estimators])

        # 确保predictions是正确的形状
        if len(predictions.shape) > 2:
            predictions = predictions.reshape(len(estimators), -1)

        # 计算预测均值和标准差
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)

        # 确保返回的sigma是一维数组
        sigma = sigma.flatten()

        return mu, sigma

    def _expected_improvement(self, X, X_sample, y_sample, xi=0.01):
        """
        计算期望改进(EI)采样函数

        参数:
            X: 待评估的点
            X_sample: 已采样的点
            y_sample: 已采样点的目标值
            xi: 探索参数

        返回:
            期望改进值
        """
        # 获取当前最优值 (y_sample 是原始尺度)
        y_best = y_sample.max() if self.maximize else y_sample.min() # Corrected for minimization
        y_best_opt = y_best if self.maximize else -y_best # Value used in calculation

        # 根据选择的方法获取预测均值和标准差 (预测时内部会处理缩放)
        if self.method == 'gp':
            # 如果有 scaler，则对 X 进行转换
            X_scaled = self.scaler.transform(X) if self.scaler else X
            mu, sigma = self.gp_model.predict(X_scaled, return_std=True)
            mu = mu.reshape(-1)  # 确保一维数组
            sigma = sigma.reshape(-1)  # 确保一维数组
            if not self.maximize: # If minimizing, GP predicts negative, convert back
                mu = -mu
        elif self.method == 'rf':
            mu, sigma = self._predict_random_forest(X) # 内部处理缩放
        elif self.method == 'bootstrap':
            mu, sigma = self._predict_bootstrap(X) # 内部处理缩放
            if not self.maximize: # If minimizing, bootstrap predicts negative, convert back
                mu = -mu
        else:
            raise ValueError(f"不支持的不确定度估计方法: {self.method}")

        # 确保sigma非负且处理零标准差情况
        sigma = np.maximum(sigma, 1e-9) # Avoid division by zero

        # 计算改进量
        if self.maximize:
            imp = mu - y_best_opt - xi
        else:
            imp = y_best_opt - mu - xi # For minimization

        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # 处理零标准差的情况 (理论上已被 np.maximum 避免, 但保留以防万一)
        ei[sigma <= 1e-9] = 0.0

        # 返回前添加展平操作
        return ei.reshape(-1)

    def _upper_confidence_bound(self, X, X_sample, y_sample, kappa=2.0):
        """
        计算置信上界(UCB)采样函数

        参数:
            X: 待评估的点
            X_sample: 已采样的点 (unused, kept for interface consistency)
            y_sample: 已采样点的目标值 (unused, kept for interface consistency)
            kappa: 探索参数

        返回:
            UCB值
        """
        # 根据选择的方法获取预测均值和标准差 (预测时内部会处理缩放)
        if self.method == 'gp':
            # 如果有 scaler，则对 X 进行转换
            X_scaled = self.scaler.transform(X) if self.scaler else X
            mu, sigma = self.gp_model.predict(X_scaled, return_std=True)
            mu = mu.reshape(-1)  # 确保一维数组
            sigma = sigma.reshape(-1)  # 确保一维数组
            if not self.maximize: # If minimizing, GP predicts negative, convert back
                mu = -mu
        elif self.method == 'rf':
            mu, sigma = self._predict_random_forest(X) # 内部处理缩放
        elif self.method == 'bootstrap':
            mu, sigma = self._predict_bootstrap(X) # 内部处理缩放
            if not self.maximize: # If minimizing, bootstrap predicts negative, convert back
                mu = -mu
        else:
            raise ValueError(f"不支持的不确定度估计方法: {self.method}")

        # 确保sigma非负
        sigma = np.maximum(sigma, 0)

        # 计算UCB (Maximize: mu + kappa * sigma, Minimize: mu - kappa * sigma)
        if self.maximize:
            ucb = mu + kappa * sigma
        else:
            ucb = mu - kappa * sigma # Lower Confidence Bound for minimization

        # 返回前添加展平操作
        return ucb.reshape(-1)

    def _probability_improvement(self, X, X_sample, y_sample, xi=0.01):
        """
        计算改进概率(PI)采样函数

        参数:
            X: 待评估的点
            X_sample: 已采样的点
            y_sample: 已采样点的目标值
            xi: 探索参数

        返回:
            改进概率值
        """
        # 获取当前最优值 (y_sample 是原始尺度)
        y_best = y_sample.max() if self.maximize else y_sample.min() # Corrected for minimization
        y_best_opt = y_best if self.maximize else -y_best # Value used in calculation

        # 根据选择的方法获取预测均值和标准差 (预测时内部会处理缩放)
        if self.method == 'gp':
            # 如果有 scaler，则对 X 进行转换
            X_scaled = self.scaler.transform(X) if self.scaler else X
            mu, sigma = self.gp_model.predict(X_scaled, return_std=True)
            mu = mu.flatten()  # 新增展平操作
            sigma = sigma.flatten()  # 新增展平操作
            if not self.maximize: # If minimizing, GP predicts negative, convert back
                mu = -mu
        elif self.method == 'rf':
            mu, sigma = self._predict_random_forest(X) # 内部处理缩放
        elif self.method == 'bootstrap':
            mu, sigma = self._predict_bootstrap(X) # 内部处理缩放
            if not self.maximize: # If minimizing, bootstrap predicts negative, convert back
                mu = -mu
        else:
            raise ValueError(f"不支持的不确定度估计方法: {self.method}")

        # 确保sigma非负且处理零标准差情况
        sigma = np.maximum(sigma, 1e-9) # Avoid division by zero

        # 计算改进概率
        if self.maximize:
            Z = (mu - y_best_opt - xi) / sigma
        else:
            Z = (y_best_opt - mu - xi) / sigma # For minimization

        pi = norm.cdf(Z)

        # 处理零标准差的情况 (理论上已被 np.maximum 避免, 但保留以防万一)
        pi[sigma <= 1e-9] = 0.0

        return pi.reshape(-1) # Ensure 1D output

    def _calculate_acquisition(self, X, X_sample, y_sample):
        """
        计算采样函数值

        参数:
            X: 待评估的点 (原始尺度)
            X_sample: 已采样的点 (原始尺度)
            y_sample: 已采样点的目标值 (原始尺度)

        返回:
            采样函数值
        """
        # Note: X_sample and y_sample are passed for context (like finding y_best)
        # The actual prediction on X happens inside the specific acquisition functions
        if self.acquisition_function == 'ei':
            return self._expected_improvement(X, X_sample, y_sample)
        elif self.acquisition_function == 'ucb':
            # UCB calculation might differ slightly depending on whether we maximize or minimize
            # The _upper_confidence_bound method handles this internally now
            return self._upper_confidence_bound(X, X_sample, y_sample)
        elif self.acquisition_function == 'pi':
            return self._probability_improvement(X, X_sample, y_sample)
        else:
            raise ValueError(f"不支持的采样函数: {self.acquisition_function}")

    def recommend_experiments(self, X, y, n_recommendations=3, max_sum=None):
        """
        推荐下一轮实验条件

        参数:
            X: 已有实验的特征矩阵 (DataFrame or NumPy array, 原始尺度)
            y: 已有实验的目标向量 (Series or NumPy array, 原始尺度)
            n_recommendations: 推荐实验数量
            max_sum: 特征和的最大值约束，如果为None则使用初始化时的值

        返回:
            推荐实验条件的DataFrame (原始尺度)
        """
        # --- START DATA ALIGNMENT ---
        if not self.feature_ranges:
            raise ValueError("在推荐实验之前必须设置特征范围。")

        feature_names = list(self.feature_ranges.keys())

        # 确保 X 是 DataFrame 以便选择列
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray) and X.shape[1] == len(feature_names):
                 X_df = pd.DataFrame(X, columns=feature_names)
            else:
                 raise TypeError("输入 X 必须是 pandas DataFrame 或列数与 feature_ranges 匹配的 NumPy 数组。")
        else:
            X_df = X

        # 只选择当前 feature_ranges 中定义的特征
        missing_features = [f for f in feature_names if f not in X_df.columns]
        if missing_features:
            raise ValueError(f"输入数据 X 缺少 feature_ranges 中定义的特征: {missing_features}")

        X_aligned = X_df[feature_names]

        # 同时对齐 y
        if isinstance(y, pd.Series):
            common_index = X_aligned.index.intersection(y.index)
            if len(common_index) != len(X_aligned):
                 pass # Or raise error/warning
            y_aligned = y.loc[common_index]
            X_aligned = X_aligned.loc[common_index]
        elif isinstance(y, np.ndarray) and len(y) == len(X_df):
            y_aligned = y[X_df.index.isin(X_aligned.index)]
            if len(y_aligned) != len(X_aligned):
                 raise ValueError("无法将 NumPy y 与 X_aligned 对齐。")
        else:
            raise TypeError("输入 y 必须是 pandas Series 或与 X 对齐的 NumPy 数组。")
        # --- END DATA ALIGNMENT ---

        # 根据选择的方法拟合/更新内部模型 (使用对齐后的数据, 内部处理缩放)
        if self.method == 'gp':
            # If the base model is GP, use it directly after fitting on aligned data
            if hasattr(self.model, 'kernel_') and 'GaussianProcessRegressor' in str(type(self.model)):
                self.gp_model = self.model
                # Fit GP on potentially scaled data
                X_fit = self.scaler.transform(X_aligned) if self.scaler else X_aligned
                y_opt_fit = y_aligned if self.maximize else -y_aligned
                self.gp_model.fit(X_fit, y_opt_fit)
            else:
                # Fit a new internal GP model (uses scaler internally if available)
                self._fit_gp(X_aligned, y_aligned)
        elif self.method == 'rf':
            if not hasattr(self.model, 'estimators_'):
                raise ValueError("使用随机森林方法时，模型必须是RandomForestRegressor类型")
            # No explicit fitting needed here for RF uncertainty, relies on base model
        elif self.method == 'bootstrap':
            # Fit bootstrap models (uses scaler internally if available)
            self._fit_bootstrap(X_aligned, y_aligned)
        else:
            raise ValueError(f"不支持的不确定度估计方法: {self.method}")

        # 定义用于搜索策略的采集函数
        # y_aligned is already on the original scale
        def acquisition_func_for_search(points_to_evaluate):
            # points_to_evaluate 可能是 DataFrame 或 numpy array (原始尺度)
            # _calculate_acquisition 需要 numpy array (原始尺度)
            if isinstance(points_to_evaluate, pd.DataFrame):
                feature_cols = list(self.feature_ranges.keys())
                missing_eval_features = [f for f in feature_cols if f not in points_to_evaluate.columns]
                if missing_eval_features:
                    raise ValueError(f"搜索策略输出缺少特征: {missing_eval_features}")
                points_array = points_to_evaluate[feature_cols].values
            elif isinstance(points_to_evaluate, np.ndarray):
                if points_to_evaluate.shape[1] != len(self.feature_ranges):
                    raise ValueError(f"搜索策略返回 NumPy 数组的特征数量不正确。预期 {len(self.feature_ranges)}, 得到 {points_to_evaluate.shape[1]}")
                points_array = points_to_evaluate
            else:
                raise TypeError(f"传递给采集函数的点类型不受支持: {type(points_to_evaluate)}")

            # Pass original scale points and original scale y_aligned
            return self._calculate_acquisition(points_array, X_aligned.values, y_aligned.values)

        # 确定使用的约束值
        constraint = max_sum if max_sum is not None else self.max_sum
        
        # 使用选定的搜索策略执行搜索
        # Search strategy operates on the original feature scale
        search_results = self.search_strategy.search(
            model=self.model, # Base model might be needed by some strategies
            acquisition_func=acquisition_func_for_search,
            n_points=n_recommendations, # Tell strategy how many points we ideally want
            maximize=self.maximize, # Pass maximize flag to strategy
            max_sum=constraint # Pass constraint to search strategy
        )

        # --- Post-processing Search Results ---
        if not isinstance(search_results, pd.DataFrame):
             raise TypeError(f"搜索策略 {type(self.search_strategy).__name__} 未返回 DataFrame")

        if 'acquisition_value' not in search_results.columns:
             raise ValueError(f"搜索策略 {type(self.search_strategy).__name__} 返回结果缺少 'acquisition_value' 列")

        feature_cols_for_pred = list(self.feature_ranges.keys())
        missing_result_features = [f for f in feature_cols_for_pred if f not in search_results.columns]
        if missing_result_features:
            raise ValueError(f"搜索策略返回结果缺少特征列: {missing_result_features}")

        # Ensure 'predicted_value' column exists, calculating if necessary
        if self.model is not None and 'predicted_value' not in search_results.columns:
            try:
                # Extract features for prediction (original scale)
                points_to_predict_original = search_results[feature_cols_for_pred].values

                # Use appropriate prediction method (handles scaling internally)
                if self.method == 'bootstrap' and self.bootstrap_models:
                    mu, _ = self._predict_bootstrap(points_to_predict_original)
                    pred_values = mu if self.maximize else -mu
                elif self.method == 'gp' and self.gp_model:
                    X_pred_scaled = self.scaler.transform(points_to_predict_original) if self.scaler else points_to_predict_original
                    mu, _ = self.gp_model.predict(X_pred_scaled, return_std=True)
                    pred_values = mu if self.maximize else -mu
                elif self.method == 'rf':
                     mu, _ = self._predict_random_forest(points_to_predict_original)
                     pred_values = mu
                else: # Fallback to base model
                     X_pred_scaled = self.scaler.transform(points_to_predict_original) if self.scaler else points_to_predict_original
                     pred_values = self.model.predict(X_pred_scaled)

                search_results['predicted_value'] = pred_values

            except ValueError as e:
                raise ValueError(f"使用内部/基础模型预测 'predicted_value' 时出错。特征可能不匹配或缩放器问题。原始错误: {e}")
            except Exception as e:
                raise RuntimeError(f"计算 'predicted_value' 时发生意外错误: {e}")

        # Return final recommendations (original scale)
        # Ensure only requested number of points are returned, even if strategy gave more
        # Sort again by acquisition value just in case strategy didn't return sorted
        # Handle maximization/minimization for sorting acquisition value
        sort_ascending = not self.maximize if self.acquisition_function == 'ucb' else False # UCB needs reverse sort for minimization
        recommendations = search_results.sort_values('acquisition_value', ascending=sort_ascending).head(n_recommendations).reset_index(drop=True)


        return recommendations

    def predict_for_features(self, feature_values):
        """
        根据给定的特征值进行预测

        参数:
            feature_values: 特征值字典，格式为 {feature_name: value}

        返回:
            预测值
        """
        if self.model is None:
            raise ValueError("模型未设置，无法进行预测")

        # 检查特征值是否完整
        if self.feature_ranges is not None:
            missing_features = [f for f in self.feature_ranges.keys() if f not in feature_values]
            if missing_features:
                raise ValueError(f"缺少以下特征的值: {', '.join(missing_features)}")

        # 创建输入数据框，并确保列顺序正确
        input_df_original = pd.DataFrame([feature_values])[list(self.feature_ranges.keys())]

        # 如果有 scaler，则对输入数据进行转换
        if self.scaler:
            if not hasattr(self.scaler, 'mean_'):
                 raise ValueError("Scaler 尚未拟合，无法转换输入特征。")
            input_df_scaled = self.scaler.transform(input_df_original)
            # 转换回 DataFrame 以保持列名，便于模型预测
            input_df_scaled = pd.DataFrame(input_df_scaled, columns=input_df_original.columns)
        else:
            input_df_scaled = input_df_original # 如果没有 scaler，则使用原始数据

        # 进行预测 - 处理模型可能是字典的情况
        if isinstance(self.model, dict) and 'name' in self.model and 'type' in self.model:
            # 如果模型是字典格式，说明是旧版本的模型存储方式，需要提示用户重新训练模型
            raise ValueError("模型格式不正确，请重新训练模型")
        else:
            # 正常情况下，模型应该是一个具有predict方法的对象
            # 使用缩放后的数据进行预测
            prediction = self.model.predict(input_df_scaled)[0]

        return prediction

    def update_with_new_data(self, original_data, new_data, features, target):
        """
        合并原始数据和新实验数据，并返回合并后的数据集

        参数:
            original_data: 原始数据DataFrame
            new_data: 新实验数据DataFrame
            features: 特征列名列表
            target: 目标列名

        返回:
            合并后的DataFrame
        """
        # 确保新数据包含所需的列
        required_columns = features + [target]
        missing_columns = [col for col in required_columns if col not in new_data.columns]

        if missing_columns:
            raise ValueError(f"新数据缺少以下列: {', '.join(missing_columns)}")

        # 只保留所需的列
        new_data_subset = new_data[required_columns].copy()

        # 合并数据
        combined_data = pd.concat([original_data, new_data_subset], ignore_index=True)

        return combined_data
